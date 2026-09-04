# Setup dari nol (fresh clone)

Repo ini tidak menyimpan bobot model, build llama.cpp, binary fork, maupun clone
pihak ketiga — semuanya di-`.gitignore` karena besar dan bisa diambil ulang.
Dokumen ini yang membuat clone di device lain tetap bisa dijalankan.

**Target hardware:** NVIDIA DGX Spark GB10, 128 GB unified memory, aarch64, CUDA 13.
Angka benchmark di `OPTIMIZATION.md` berlaku untuk kombinasi ini; hardware lain
akan berbeda.

## Prasyarat

```bash
aria2c --version        # untuk semua download (resumable, size-verified)
jq --version            # bench.sh
nvidia-smi              # driver; diuji pada 580.173.02
docker --version        # hanya untuk jalur vLLM
docker run --rm --gpus all <image> nvidia-smi -L    # pastikan GPU terlihat dari container
```

## Jalur A — llama.cpp + MTP (REKOMENDASI, ~110 GB)

Ini stack yang menang di benchmark: decode 37.8 tok/s, RAM 88 GiB, start 45 detik.

```bash
# 1. Model utama, 104 GiB, 4 part (~3 jam @ 11 MB/s)
./download-parts.sh

# 2. MTP draft head, 3.85 GiB
./download-mtp.sh

# 3. Binary fork Unsloth, 223 MB -- WAJIB untuk MTP
./download-fork.sh

# 4. Jalankan
./stack.sh start llamacpp        # port 18080, API OpenAI-compatible
./stack.sh status
./stack.sh stop
```

**Kenapa harus fork, bukan llama.cpp upstream:** upstream menolak MTP head Unsloth
karena `output_hc_norm.weight` tidak ada di file head (upstream mewajibkan set tensor
`qwen4exp` lengkap). Tanpa MTP, decode turun dari 37.8 ke ~24 tok/s. Detail di
`OPTIMIZATION.md`.

### Opsional: build llama.cpp upstream sendiri

Hanya perlu kalau ingin membandingkan dengan upstream (tanpa MTP):

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -B llama.cpp/build-new -S llama.cpp \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=OFF
cmake --build llama.cpp/build-new -j8        # ~4 menit di GB10
BUILD=build-new ./stack.sh start llamacpp    # tanpa --mtp
```

`CMAKE_CUDA_ARCHITECTURES=native` penting: hanya sm_121 yang dikompilasi, bukan fatbin
multi-arsitektur. Itu yang membuat build 4 menit, bukan 40.

## Jalur B — vLLM NVFP4 (opsional, ~150 GB)

Lebih cepat 1.44× hanya pada panggilan cold; kalah saat prefix cache hit, dan makan
112 GiB (sisa ~8 GiB). Lihat perbandingan lengkap di `OPTIMIZATION.md`.

```bash
git clone https://github.com/blazux/qwen3.8-Flash-DGX.git vllm-dgx
cd vllm-dgx && docker build -t qwen38-flash-dgx . && cd ..     # base image ~23 GB
# token HF opsional tapi menghindari rate-limit anonim:
#   printf '%s' 'hf_xxx' > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
vllm-dgx/scripts/download-weights.sh                           # 126 GB, ~3 jam
./stack.sh start vllm                                          # port 18300, load ~14 menit
```

## Aturan operasional (penting)

Unified memory 121.7 GiB dibagi CPU+GPU. **Jangan pernah menjalankan dua stack
sekaligus, atau satu stack bersama Ollama yang sedang memegang model** — itu
menyebabkan `NVRM NV_ERR_NO_MEMORY` → stall → kernel panic watchdog (dialami
2026-09-01). `stack.sh` menolak start kalau kondisi itu terdeteksi, tapi jangan
mem-bypass-nya secara manual.

```bash
./stack.sh status        # cek sebelum apa pun
```

`stack.sh` menjalankan llama-server dengan `-ub 256`. Tanpa itu backend CUDA (fork
maupun upstream) bisa abort di `cublasGemmEx` pada batch prefill tertentu — satu prompt
HumanEval mereproduksinya setiap kali, sweep menemukan ukuran 367 dan 512 ikut crash.
Kalau memanggil `serve.sh` langsung, set `UBATCH=256`. Detail di `OPTIMIZATION.md`.

## Benchmark ulang

Prompt benchmark dibaca dari source `llama.cpp` asli supaya identik lintas run dan
lintas mesin. `llama.cpp/` di-gitignore, jadi di fresh clone perlu diambil dulu:

```bash
git clone https://github.com/ggml-org/llama.cpp.git      # tidak perlu di-build
```

```bash
./bench-code.sh <label> 3                    # via /completion llama.cpp
python3 bench-stream.py <port> <label> 3     # streaming, jalan untuk kedua stack
HOST=tailscale API_KEY=<key> python3 bench-stream.py 18080 remote 3   # dari jauh
./cache-ratio.py -v                          # rasio cache hit dari runs/serve-current.log
tmp/eval-venv/bin/python bench-accuracy.py 18080 <label> --concurrent 1   # akurasi (lihat bawah)
```

Semua script bisa dipanggil dari direktori mana pun — masing-masing pindah ke
direktorinya sendiri lebih dulu.

`bench-stream.py` yang dipakai untuk membandingkan lintas stack: mengukur dari sisi
klien dan membaca jumlah token dari `usage.completion_tokens`. Jangan bandingkan
angka dari `bench.sh`/`bench-code.sh` (timings sisi server llama.cpp) dengan angka
vLLM — definisinya berbeda. Tiga cacat metodologi yang pernah terjadi tercatat di
`OPTIMIZATION.md`.

`cache-ratio.py` menjawab pertanyaan yang ditinggalkan `OPTIMIZATION.md`: berapa persen
panggilan nyata yang cold. Ia membaca log llama-server (bukan benchmark), merekonstruksi
token yang di-cache per request dari `prompt eval` vs `n_tokens` saat release, dan
membandingkannya dengan titik impas 12%. Jalankan setelah sesi coding agent sungguhan,
bukan setelah benchmark — prompt benchmark sengaja berbeda tiap run.

### Akurasi (`bench-accuracy.py`)

Butuh venv sekali pakai di `tmp/` (tidak di-track):

```bash
uv venv tmp/eval-venv --python 3.12
VIRTUAL_ENV=$PWD/tmp/eval-venv uv pip install "lm_eval[api]" evalplus
```

Dataset (GSM8K, HumanEval+) diunduh ke `tmp/hf-cache` dan `tmp/evalplus-cache`, bukan ke
`~/.cache/huggingface` yang dimiliki root oleh container vLLM. Jalankan dengan
`--concurrent 1` di llama.cpp + MTP (upstream #28286: kontaminasi antar slot); di vLLM
`--concurrent 4` aman. GSM8K 300 soal + HumanEval+ 164 ≈ 50 menit per backend. Baca skor
`acc_final_para`, bukan `acc_flexible_extract` — alasannya di `OPTIMIZATION.md`.

## Menyambungkan coding agent / harness

Kedua stack menyediakan **API OpenAI-compatible**, jadi harness apa pun yang bisa
diarahkan ke base URL kustom bisa dipakai. Diuji pada llama.cpp fork (2026-09-04):

| Kemampuan | Status |
|---|---|
| `GET /v1/models` | ok — model id `qwen3.8-flash-next` (via `--alias`) |
| `POST /v1/chat/completions` | ok |
| `POST /v1/completions` | ok |
| Streaming SSE | ok |
| **Tool / function calling** | ok — balas `tool_calls` + `finish_reason: "tool_calls"` |

Konfigurasi klien:

```
OPENAI_BASE_URL = http://127.0.0.1:18080/v1     # vLLM: 18300
OPENAI_API_KEY  = apa saja (tidak diperiksa)
model           = qwen3.8-flash-next
```

**Model ini reasoning model.** Jawaban dipecah jadi dua field: `reasoning_content`
(proses berpikir) dan `content` (jawaban final). Harness yang hanya membaca
`content` tetap bekerja normal — hanya tidak melihat proses berpikirnya. Kalau
harness kebingungan atau kamu ingin hemat token, matikan dengan
`--reasoning off` atau `--no-reasoning-preserve` di `serve.sh`.

**Context default 131072** (bukan 32768 seperti versi awal, bukan pula 262144 native).
KV cache q8_0: ~6.4 GiB di 128k vs ~12.8 GiB di 262k. Ubah dengan
`CTX=262144 ./stack.sh start llamacpp` — RAM naik dari 91 ke ~97 GiB.

### Pi (https://pi.dev) — konfigurasi siap pakai

`pi-models.json` di direktori ini berisi dua provider (llama.cpp 18080, vLLM 18300).
Salin atau gabungkan ke `~/.pi/agent/models.json`, lalu pilih lewat `/model`:

```bash
mkdir -p ~/.pi/agent
cp pi-models.json ~/.pi/agent/models.json      # HATI-HATI: jangan timpa config yang sudah ada
```

`apiKey` diisi `"$QWEN38_API_KEY"` — sintaks Pi untuk membaca environment variable
(`$VAR`, `${VAR}`, atau `!command`). Bentuk `{env:VAR}` yang dipakai revisi lama TIDAK
dikenal Pi: placeholder-nya dikirim apa adanya dan server membalas 401. Export
variabelnya di shell yang sama dengan tempat `pi` dijalankan. Kalau server jalan tanpa
`API_KEY`, isi variabelnya sembarang string non-kosong: Pi menyembunyikan model yang
belum punya key dari `/model`.

Kalau laptop memakai NVIDIA Sync (bukan aplikasi Tailscale), `HOST=tailscale` tidak bisa
dijangkau dari laptop itu — node Tailscale Sync tertanam di dalam aplikasi, OS laptop
tidak punya interface `100.x`. Pakai port-forward Sync: Settings → Custom → Add New,
port 18080, launch script `cd <repo> && API_KEY=<key> ./stack.sh start llamacpp`,
`HOST` biarkan default. Setelah dibuat, **klik entry-nya** di daftar Custom sampai
titik statusnya hijau — Add hanya menyimpan definisi, forward baru aktif setelah diklik.
`baseUrl` di laptop jadi `http://127.0.0.1:18080/v1`. Detail di
README "Reaching the server from another machine". Diverifikasi 2026-09-04.

Setiap field disetel dari hasil probe ke server yang benar-benar jalan (2026-09-04),
bukan dari dokumentasi:

| Yang diuji | Hasil | Konsekuensi di models.json |
|---|---|---|
| role `developer` | HTTP 200 | `supportsDeveloperRole` biarkan default (true) |
| `reasoning_effort` | HTTP 200 | `supportsReasoningEffort` biarkan default (true) |
| `max_tokens` / `max_completion_tokens` | dua-duanya 200 | `maxTokensField` tidak perlu diset |
| `stream_options.include_usage` | `usage` terkirim | `supportsUsageInStreaming` true |
| bentuk thinking | `reasoning_content` | `"thinkingFormat": "deepseek"` |
| tool calling | `tool_calls` + `finish_reason` | jalan tanpa penyesuaian |

Catatan: llama.cpp *menerima* `reasoning_effort` tanpa error, tapi kemungkinan
mengabaikannya. Untuk Pi itu cukup — yang penting request tidak ditolak.

Bonus dari probe: `usage.prompt_tokens_details.cached_tokens` ikut dilaporkan, jadi
kamu bisa memantau langsung berapa banyak prompt yang kena prefix cache — angka yang
menentukan apakah llama.cpp atau vLLM lebih cocok (titik impas 12% cold, lihat
`OPTIMIZATION.md`).
