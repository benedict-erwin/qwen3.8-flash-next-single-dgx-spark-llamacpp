# Riset: Qwen3.8-Flash-Next di DGX Spark GB10 128GB

> **Historical working notes (2026-08-31, in Indonesian):** the initial survey of quant
> candidates and serving stacks that led to the choice documented in `README.md` and
> `OPTIMIZATION.md`. Kept as written; the current, English record of what was actually
> measured is `README.md`, with the full method in `OPTIMIZATION.md`.

> Tanggal: 2026-08-31. Target user: muat "agak longgar" di 128GB unified memory, retensi akurasi > 90%, tps setinggi mungkin; bonus varian abliterated/zero-refusal yang tetap akurat.

## Fakta model

- **Qwen3.8-Flash-Next** (rilis 2026-08-26): open-weight preview arsitektur Qwen4 (`qwen4exp`). MoE **125B total / 6B aktif**, plus **PLE n-gram embedding table 51B** dan **MTP head 4B** → ~180B stored, BF16 ~355GB, FP8 resmi ~186GB. Native context 262K (1M via YaRN). GPQA Diamond 91.7; klaim menang atas Opus 4.6 Max di SWE-bench Pro (62.5 vs 53.4).
- **Kunci untuk GB10:** PLE n-gram table (51B) punya pola akses random dan TIDAK harus resident — bisa di-stream dari NVMe (llama.cpp `-ot "per_layer_token_embd.weight=CPU"`, atau mmap di vLLM patch). Ini yang bikin model 180B muat longgar di 128GB.
- **Support:** llama.cpp PR #27742 (arsitektur `qwen4exp`) **sudah merged ke master**; Ollama versi terbaru sudah bisa (library `qwen3.8-flash-next` tersedia, termasuk `125b-a6b-nvfp4`). Build lama tidak bisa load sama sekali.
- Retensi akurasi quant (Unsloth, "top-1% accuracy" vs BF16): UD-Q4_K_XL 111.3GB = **92.3%**; UD-IQ4_XS 93.7GB = 89.6%; UD-Q3_K_XL 90GB = 88.3%; UD-Q5_K_XL 158GB = 93.7% (tidak muat). PLE minimal 4-bit — quant lebih agresif merusak.

## Kandidat

### A. UD-Q4_K_XL (Unsloth) + PLE streamed dari NVMe — llama.cpp/Ollama
111.3GB file, resident jauh lebih kecil karena PLE 51B off-RAM. Retensi **92.3%** (satu-satunya yang >90% dan muat). Terukur di GB10: decode **~25 tok/s**, prefill 77 tok/s, ctx hingga 700K–1M dengan KV 4-bit.
- ➕ Retensi tertinggi yang muat; jalur paling standar (llama.cpp master, Ollama); ctx terpanjang.
- ➖ Prefill lambat (TTFT panjang di prompt besar); load ~1.5 menit; butuh flag offload manual.

### B. NVFP4 + vLLM patched (repo blazux/qwen3.8-Flash-DGX) — tps terbaik
Resident **~76 GiB** (hybrid fp8: 69 GiB) — paling longgar. Decode **26 tok/s** (MTP=2), hybrid **31 tok/s**; prefill **1.500–2.000 tok/s**; KV ~580–630K token; YaRN 500K tervalidasi needle-414K. NVFP4 native di Blackwell; klaim quality identik dengan hybrid.
- ➕ Kombinasi terbaik: longgar + decode tercepat + prefill 20–25× opsi A; prefix-cache TTFT 1.4s.
- ➖ Stack custom (docker + patch vLLM), bukan Ollama; kernel sparse-attention stock non-deterministik (fix `EXACT_TOPK=1` biaya 10–40% prefill); single-model (85% memory); retensi akurasi NVFP4 belum diukur pihak ketiga `[Belum Terverifikasi]`.

### C. Baekpica Mixed-Quant SSD-PLE (Q5/Q6) — llama.cpp-based custom
Q5: resident **77.5 GiB**; Q6: 91 GiB; PLE **BF16 di SSD** (O_DIRECT, zero quality loss di embedding). Terukur GB10: decode 17.6 → **28.65 tok/s dengan embedded MTP**; prefill 471 tok/s; 262K full-context tervalidasi. Q5 vs Q6: -1.4% NLL saja.
- ➕ Longgar + akurasi backbone Q5/Q6 + PLE tanpa quant; MTP +21% decode; multimodal jalan.
- ➖ TTFT 17s; serving stack niche (bukan Ollama); komunitas kecil, verifikasi terbatas.

### D. UD-IQ4_XS (Unsloth) full-resident — paling simpel
93.7GB, muat utuh di RAM tanpa trik SSD, langsung Ollama.
- ➕ Setup termudah, no offload.
- ➖ Retensi **89.6%** — meleset tipis dari target 90%; headroom KV ~25GB (ctx terbatas).

## Kandidat abliterated (zero refusal)

### E. cygnal Uncensored-IQ4XS-NGQ4 — abliterated, muat full di RAM ⭐ paling praktis
**98.4GB total** (IQ4_XS bulk + output Q6_K + **n-gram Q4_0**, 5.61 BPW), dari checkpoint abliterasi orcarouter. HumanEval **82.3%** / HumanEval+ 78.0. Decode 20.5–21.7 tok/s (terukur di Ryzen AI Max 395, kelas bandwidth mirip GB10 `[Inferensi]`).
- ➕ Muat utuh di 128GB tanpa offload; akurasi coding terukur; llama.cpp master sudah support.
- ➖ Headroom tinggal ~25GB (KV terbatas, "longgar"-nya pas-pasan); n-gram di Q4_0 = batas bawah yang aman.

### F. orcarouter Uncensored-GGUF (Q4_K_M ~110GB) + PLE offload — abliterated referensi
Sumber abliterasi utama. Refusal **64–100% → ~0–3.3%**, kapabilitas **±2 pts** dari baseline (MMLU-Pro, GSM8K, CMMLU); vision + tool calling jalan. Tersedia juga di ollama.com (`orcarouter/Qwen3.8-Flash-Next-Uncensored`). Perlu trik PLE offload ala opsi A supaya longgar.
- ➕ Klaim zero-refusal + retensi terdokumentasi paling lengkap; banyak pilihan quant (IQ2–Q6_K).
- ➖ Angka refusal/akurasi = self-reported `[Belum Terverifikasi]`; PLE fallback Q8_0 di quant tinggi (file besar).

### G. mazinb Uncensored-NVFP4 — abliterated untuk vLLM
Jalan di GB10 (121 GiB + swapfile ~100 GiB untuk PLE): **17 tok/s** single, 57 tok/s @4 concurrent.
- ➕ Satu-satunya abliterated jalur NVFP4/vLLM; throughput concurrent ok.
- ➖ Lebih lambat dari B; setup swapfile ribet; tanpa benchmark akurasi sama sekali.

## Rekomendasi awal

- **Butuh tps & longgar maksimal (non-abliterated):** **B** (blazux NVFP4+vLLM) — kalau mau tetap di Ollama: **A**.
- **Abliterated:** **E** (cygnal) paling praktis dan terukur; **F** kalau mau kontrol quant + angka refusal terdokumentasi.
- Kompromi tunggal "semua kriteria": **A** (>90% terbukti, 25 tok/s, ctx 1M) — abliterated menyusul via F dengan setup yang sama.

## Optimasi lanjutan untuk recipe sendiri (di atas opsi A)

1. **MTP speculative decoding — gain terbesar, gratis.** GGUF Unsloth menyimpan tensor `blk.*.nextn.*`; llama.cpp (PR #22673) bisa memakainya via `--spec-type draft-mtp` + `--spec-draft-n-max 2..3`. Terukur: +21% decode di Flash-Next (Baekpica embedded-MTP), +33–145% di keluarga Qwen3.8-27B. Estimasi A: 25 → ~30 tok/s `[estimate]`.
2. **KV cache quant + flash attention**: KV q8_0 (aman) atau q4 (ctx 700K–1M di GB10, ada degradasi kualitas long-context).
3. **PLE placement 3 tingkat**: sisa RAM → CPU-pinned → SSD O_DIRECT (resep Baekpica). Default opsi A: `-ot "per_layer_token_embd.weight=CPU"`.
4. **Quant mix sendiri** (`llama-quantize` + `--tensor-type` overrides + imatrix dari data domain sendiri, mis. ID/EN campur): resep region-aware Baekpica (edge layers & shared expert presisi lebih tinggi, routed experts tengah lebih rendah, MTP head Q8_0, PLE tidak di bawah 4-bit) terbukti resident 77.5 GiB dengan -1.4% NLL. Butuh source Q8_0 188GB atau BF16 355GB (download besar — konfirmasi dulu).
5. **YaRN** untuk context >262K native.
6. Requantize checkpoint BF16 uncensored (orcarouter) dengan imatrix sendiri = jalan menuju "UD-style uncensored" yang belum ada di publik — degradasi gabungan wajib diukur sendiri.

## Sumber

- https://huggingface.co/Qwen/Qwen3.8-Flash-Next · https://llm-stats.com/blog/research/qwen3.8-flash-next-launch · https://www.datacamp.com/blog/qwen3-8-flash-next
- https://unsloth.ai/docs/models/qwen3.8-next · https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
- https://forums.developer.nvidia.com/t/qwen3-8-flash-next-ud-q4-k-xl-gguf-on-dgx-spark-with-llama-cpp-gpu-experts-ple-n-gram-table-streamed-from-disk-25-tok-s-up-to-1m-context/381720
- https://github.com/blazux/qwen3.8-Flash-DGX
- https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF
- https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF · https://huggingface.co/cygnal/Qwen3.8-Flash-Next-Uncensored-IQ4XS-NGQ4-GGUF · https://huggingface.co/mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4
- https://github.com/ggml-org/llama.cpp/pull/27742 · https://ollama.com/library/qwen3.8-flash-next
