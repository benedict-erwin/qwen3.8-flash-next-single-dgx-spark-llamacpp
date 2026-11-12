// Cold-L2 GEMV benchmark through the ggml CUDA backend: N distinct weight copies (>= 256 MB in total)
// are multiplied in one graph so every launch streams its weights from DRAM, like a decode step does.
//
//   mmvq_bench q8_0 K M n [copies]            dense: y_i = W_i[K->M] * x[K, n]
//   mmvq_bench q4_K K M n moe E U [copies]    experts: mul_mat_id over E experts, U used per token
//
// Prints us per op and GB/s against the weight bytes actually touched.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static ggml_type parse_type(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * name = ggml_type_name((ggml_type) t);
        if (name && strcmp(name, s) == 0) return (ggml_type) t;
    }
    fprintf(stderr, "unknown type %s\n", s); exit(1);
}

static void fill_quantized(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        std::vector<float> f(n);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (auto & v : f) v = d(rng);
        ggml_backend_tensor_set(t, f.data(), 0, n*sizeof(float));
        return;
    }
    // quantized weights: random bytes (values are irrelevant for a streaming benchmark; keeps setup fast)
    std::vector<uint8_t> q(ggml_nbytes(t));
    uint32_t st = rng();
    for (auto & b : q) { st = st*1664525u + 1013904223u; b = st >> 24; }
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: see source\n"); return 1; }
    const ggml_type type = parse_type(argv[1]);
    const int64_t K = atoll(argv[2]), M = atoll(argv[3]), n = atoll(argv[4]);
    bool moe = argc > 5 && strcmp(argv[5], "moe") == 0;
    int64_t E = 1, U = 1;
    int copies = 0;
    if (moe) { E = atoll(argv[6]); U = atoll(argv[7]); if (argc > 8) copies = atoi(argv[8]); }
    else if (argc > 5) copies = atoi(argv[5]);

    const size_t row_bytes = ggml_row_size(type, K);
    const size_t w_bytes   = moe ? row_bytes*M*U*n : row_bytes*M;   // bytes touched per op (moe: upper bound, distinct experts)
    const size_t w_alloc   = row_bytes*M*E;
    if (copies <= 0) copies = (int) std::max<size_t>(1, std::min<size_t>(moe ? 8 : 64, (256u<<20) / w_bytes + 1));

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "no CUDA backend\n"); return 1; }

    const int gap0 = (getenv("GAP") ? atoi(getenv("GAP")) : 0) + (getenv("COMPUTE") ? atoi(getenv("COMPUTE")) : 0);
    ggml_init_params ip = { ggml_tensor_overhead()*(size_t)((4 + gap0)*copies + 16) + ggml_graph_overhead_custom((4 + gap0)*copies + 8, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x = moe ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, n) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
    std::vector<ggml_tensor *> W(copies), ids(copies), outs(copies);
    for (int i = 0; i < copies; ++i) {
        W[i] = moe ? ggml_new_tensor_3d(ctx, type, K, M, E) : ggml_new_tensor_2d(ctx, type, K, M);
        if (moe) ids[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, n);
    }
    const int gap = getenv("GAP") ? atoi(getenv("GAP")) : 0;   // tiny dependent ops before each GEMV
    ggml_tensor * g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, (4 + gap)*copies + 8, false);
    // COMPUTE=N: N chained f32 matmuls [1024x1024]x[1024x1024] (L2-resident, SM-heavy, little DRAM) before each GEMV
    const int ncomp = getenv("COMPUTE") ? atoi(getenv("COMPUTE")) : 0;
    ggml_tensor * ca = ncomp ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024) : nullptr;
    ggml_tensor * cb = ncomp ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024) : nullptr;
    for (int i = 0; i < copies; ++i) {
        for (int j = 0; j < ncomp; ++j) { cb = ggml_mul_mat(ctx, ca, cb); ggml_build_forward_expand(gf, cb); }
        for (int j = 0; j < gap; ++j) { g = ggml_scale(ctx, g, 1.0001f); ggml_build_forward_expand(gf, g); }
        outs[i] = moe ? ggml_mul_mat_id(ctx, W[i], x, ids[i]) : ggml_mul_mat(ctx, W[i], x);
        ggml_build_forward_expand(gf, outs[i]);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc failed (%zu MB)\n", (w_alloc*copies)>>20); return 1; }

    std::mt19937 rng(1234);
    fill_quantized(x, rng);
    fill_quantized(g, rng);
    if (ca) { fill_quantized(ca, rng); }
    for (int i = 0; i < copies; ++i) {
        fill_quantized(W[i], rng);
        if (moe) {
            std::vector<int32_t> id(U*n);
            for (int64_t t = 0; t < n; ++t) {                     // U distinct experts per token
                std::vector<int32_t> perm(E); for (int64_t e = 0; e < E; ++e) perm[e] = e;
                std::shuffle(perm.begin(), perm.end(), rng);
                for (int64_t u = 0; u < U; ++u) id[t*U + u] = perm[u];
            }
            ggml_backend_tensor_set(ids[i], id.data(), 0, id.size()*sizeof(int32_t));
        }
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    for (int i = 0; i < 5; ++i) { ggml_backend_graph_compute(backend, gf); }
    ggml_backend_synchronize(backend);
    const int iters = std::max(3, (int) (2000000000ull / (w_bytes*copies) ));   // ~2 GB streamed... at least 3 iters
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { ggml_backend_graph_compute(backend, gf); }
    ggml_backend_synchronize(backend);
    auto t1 = std::chrono::high_resolution_clock::now();
    const double us_op = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters / copies;
    printf("%-5s %s K=%lld M=%lld n=%lld%s copies=%d: %8.1f us/op  %6.1f GB/s  (%.1f MB/op)\n",
           ggml_type_name(type), moe ? "moe" : "dense", (long long) K, (long long) M, (long long) n,
           moe ? (" E=" + std::to_string(E) + " U=" + std::to_string(U)).c_str() : "",
           copies, us_op, w_bytes / us_op / 1e3, w_bytes / 1048576.0);

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
