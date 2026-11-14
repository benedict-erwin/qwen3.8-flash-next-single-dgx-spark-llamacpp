// Can a cold GEMV overlap the GatedDeltaNet core on a second CUDA stream? Two ggml CUDA backends
// (two streams, one process): one thread runs ncomp GDN core ops (this model's shapes, each on its
// own 3 MB state), the other streams cold Q8_0 GEMVs. Rates alone vs together give the answer.
//   overlap_bench K M n ncomp [secs]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>
using clk = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }
static void fill(ggml_tensor * t, uint32_t seed, bool f32) {
    if (f32) { std::vector<float> f(ggml_nelements(t)); std::mt19937 r(seed); std::uniform_real_distribution<float> d(-0.03f, 0.03f);
        for (auto & v : f) v = d(r); ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t)); return; }
    std::vector<uint8_t> q(ggml_nbytes(t)); uint32_t st = seed; for (auto & b : q) { st = st*1664525u + 1013904223u; b = st >> 24; }
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}
struct Job { ggml_backend_t be; ggml_context * ctx; ggml_cgraph * gf; ggml_gallocr_t ga; int ops; };
static Job make_gemv(int64_t K, int64_t M, int64_t n, int copies) {
    Job j; j.be = ggml_backend_cuda_init(0); j.ops = copies;
    ggml_init_params ip = { ggml_tensor_overhead()*(size_t)(4*copies + 16) + ggml_graph_overhead_custom(4*copies + 8, false), nullptr, true };
    j.ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(j.ctx, GGML_TYPE_F32, K, n);
    std::vector<ggml_tensor *> W(copies);
    for (auto & w : W) w = ggml_new_tensor_2d(j.ctx, GGML_TYPE_Q8_0, K, M);
    j.gf = ggml_new_graph_custom(j.ctx, 4*copies + 8, false);
    for (auto w : W) ggml_build_forward_expand(j.gf, ggml_mul_mat(j.ctx, w, x));
    ggml_backend_alloc_ctx_tensors(j.ctx, j.be);
    fill(x, 1, true); uint32_t s = 100; for (auto w : W) fill(w, s++, false);
    j.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(j.be)); ggml_gallocr_alloc_graph(j.ga, j.gf);
    return j;
}
static Job make_busy(int ncomp, int64_t n) {   // ncomp real GDN core ops, each on its own 3 MB state
    Job j; j.be = ggml_backend_cuda_init(0); j.ops = ncomp;
    ggml_init_params ip = { ggml_tensor_overhead()*(size_t)(4*ncomp + 16) + ggml_graph_overhead_custom(4*ncomp + 8, false), nullptr, true };
    j.ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(j.ctx, GGML_TYPE_F32, 128, 16, n, 1);
    ggml_tensor * k = ggml_new_tensor_4d(j.ctx, GGML_TYPE_F32, 128, 16, n, 1);
    ggml_tensor * v = ggml_new_tensor_4d(j.ctx, GGML_TYPE_F32, 128, 48, n, 1);
    ggml_tensor * g = ggml_new_tensor_4d(j.ctx, GGML_TYPE_F32, 1, 48, n, 1);
    ggml_tensor * be = ggml_new_tensor_4d(j.ctx, GGML_TYPE_F32, 1, 48, n, 1);
    std::vector<ggml_tensor *> st(ncomp);
    for (auto & s : st) s = ggml_new_tensor_4d(j.ctx, GGML_TYPE_F32, 128, 128, 48, 1);
    j.gf = ggml_new_graph_custom(j.ctx, 4*ncomp + 8, false);
    for (auto s : st) ggml_build_forward_expand(j.gf, ggml_gated_delta_net(j.ctx, q, k, v, g, be, s, 1));
    ggml_backend_alloc_ctx_tensors(j.ctx, j.be);
    fill(q, 7, true); fill(k, 8, true); fill(v, 9, true); fill(g, 10, true); fill(be, 11, true);
    uint32_t sd = 50; for (auto s : st) fill(s, sd++, true);
    j.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(j.be)); ggml_gallocr_alloc_graph(j.ga, j.gf);
    return j;
}
static double run(Job & j, double secs, std::atomic<bool> * go) {   // returns us per op
    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(j.be, j.gf);
    ggml_backend_synchronize(j.be);
    if (go) while (!go->load()) {}
    long it = 0; double t0 = now(), t1;
    do { ggml_backend_graph_compute(j.be, j.gf); ggml_backend_synchronize(j.be); ++it; t1 = now(); } while (t1 - t0 < secs);
    return (t1 - t0)*1e6 / it / j.ops;
}
int main(int argc, char ** argv) {
    const int64_t K = atoll(argv[1]), M = atoll(argv[2]), n = atoll(argv[3]); const int ncomp = atoi(argv[4]);
    const double secs = argc > 5 ? atof(argv[5]) : 3.0;
    const size_t wb = ggml_row_size(GGML_TYPE_Q8_0, K)*M; const int copies = (int) ((256u<<20)/wb + 1);
    Job g = make_gemv(K, M, n, copies), b = make_busy(ncomp, n);
    const double g0 = run(g, secs, nullptr), b0 = run(b, secs, nullptr);
    std::atomic<bool> go(false); double g1 = 0, b1 = 0;
    std::thread tg([&]{ g1 = run(g, secs, &go); }), tb([&]{ b1 = run(b, secs, &go); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300)); go = true; tg.join(); tb.join();
    printf("K=%lld M=%lld n=%lld: gemv alone %.1f us (%.0f GB/s), busy alone %.1f us/op | together: gemv %.1f us (%.0f GB/s), busy %.1f us/op | overlap factor %.2f (1 = time-sliced, 2 = free)\n",
        (long long) K, (long long) M, (long long) n, g0, wb/g0/1e3, b0, g1, wb/g1/1e3, b1, g0/g1 + b0/b1);
    return 0;
}
