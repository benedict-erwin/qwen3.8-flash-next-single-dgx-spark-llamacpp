# Recipe optimal Qwen3.8-Flash-Next di DGX Spark GB10

> Riset lanjutan 2026-09-02. Menggantikan sebagian kesimpulan `RESULTS.md` (lihat "Koreksi" di bawah).
> Semua angka ukuran tensor di dokumen ini diparse langsung dari GGUF lokal
> (`models/UD-Q4_K_XL/*.gguf`), bukan dari model card.

> ## ⚠️ HASIL EKSEKUSI 2026-09-03 — beberapa klaim di bawah TERBANTAH
>
> Rencana di dokumen ini sudah dieksekusi sampai langkah 5. Ringkasnya: **tidak ada
> satu pun jalur yang menghasilkan percepatan.** Detail di bagian "Hasil eksekusi"
> di bawah. Yang perlu dibaca dengan koreksi:
>
> - **"+3-10% dari update build"** → sebenarnya **+0.4%** (dalam noise). Angka
>   +24.7% yang sempat kulaporkan adalah artefak metode benchmark, bukan speedup.
> - **"MTP tersedia sebagai sidecar 2.60 GB"** → benar bahwa filenya ada, **salah**
>   bahwa upstream bisa memakainya. Kedua varian ditolak `llama.cpp` 0ba6499.
> - **"Proyeksi 31-39 tok/s"** → belum terbukti dan saat ini terblokir.
> - **"origin/master sudah menyambungkan MTP ke qwen4exp"** → graph-nya ada, tapi
>   loader untuk draft head terpisah tidak. Grep 35 referensi nextn/mtp tidak cukup
>   sebagai bukti; seharusnya diuji, bukan disimpulkan.
> - **Efisiensi roofline** yang benar: **57%** (23.71 / 41.4), bukan 51% atau 66%.

## TL;DR

1. **MTP ternyata TERSEDIA untuk model yang sudah kita punya.** Kesimpulan lama ("MTP gagal, harus ganti GGUF, download besar lagi") **salah**. Unsloth merilis MTP head sebagai **file sidecar terpisah 2.60 GB** yang dipasang di samping UD-Q4_K_XL kita yang sekarang — tidak perlu download ulang 104 GB.
2. **llama.cpp kita ketinggalan 40 commit**, dan `origin/master` **sudah** menyambungkan MTP ke arsitektur `qwen4exp`. Waktu kita tes 2026-09-01, dukungan itu memang belum ada di build kita. Cukup `git pull` + rebuild.
3. **Decode kita sudah mentok memory bandwidth, bukan salah konfigurasi.** Efisiensi kita 51–53% dari roofline — sama dengan sistem GB10 lain yang dilaporkan publik. Artinya tidak ada "flag ajaib" yang tersisa; satu-satunya cara naik signifikan adalah **mengurangi byte per token** (quant lebih kecil) atau **menaikkan token per byte** (speculative decoding).
4. **Rekomendasi: recipe gabungan** — UD-Q4_K_XL (yang sudah ada) + MTP sidecar + `-lm mmap` + llama.cpp terbaru. Proyeksi **31–39 tok/s** `[estimate]` dengan retensi 92.3% tetap, resident turun dari ~104 → **~77 GiB**.

---

## 1. Batas fisik: di mana sebenarnya kita berdiri

### Anatomi model (parse langsung dari GGUF lokal)

| Kelompok tensor | GiB | % file | Dibaca per token? |
|---|---|---|---|
| MoE experts (`*_exps`) | 71.73 | 69.2% | **Tidak** — hanya 10 dari 512 expert |
| PLE (`per_layer_token_embd`) | 26.82 | 25.9% | Tidak — cuma lookup baris |
| attn / ssm / norm | 3.86 | 3.7% | **Ya, penuh** |
| `token_embd` + `output` | 1.26 | 1.2% | Sebagian (`output` penuh) |
| **TOTAL** | **103.68** | | |

Validasi: total tensor 103.68 GiB = ukuran file di disk 103.69 GiB. PLE terukur **4.500 bit/elemen** untuk 51,2 miliar elemen — cocok dengan angka "26.82 GB lookup table" yang beredar di diskusi llama.cpp.

### Roofline decode

Byte yang harus dibaca per token = attn/ssm penuh + 10/512 experts + output head:

```
attn/ssm/norm (penuh)     3.86 GiB   63%
experts aktif (10/512)    1.40 GiB   23%
output head               0.63 GiB   10%
--------------------------------------------
per token                 5.90 GiB = 6.34 GB
```

GB10 = 273 GB/s → **roofline ≈ 43 tok/s**.

| Konfigurasi | Ukuran | Terukur | % roofline |
|---|---|---|---|
| UD-Q4_K_XL — kita, 2026-09-01 | 103.69 GiB | 21.8 tok/s | **51%** |
| UD-Q4_K_XL — forum NVIDIA | 103.69 GiB | 25.0 tok/s | 58% |
| UD-IQ1_S — kubesimplify (tg128) | 67.55 GiB | 34.5 tok/s | 54% |

**Ini temuan terpenting dari riset ini.** Dua quant berbeda, di hardware yang sama, jatuh di efisiensi bandwidth yang hampir identik (51–54%). Rasio kecepatan 34.5/21.8 = 1.58 hampir persis rasio ukuran 103.69/67.55 = 1.54.

Artinya: **setup kita tidak salah konfigurasi.** Angka 34.5 tok/s yang beredar itu bukan "recipe lebih baik", itu cuma **quant yang jauh lebih kecil dan lebih rusak** (IQ1_S = 3.28 bpw efektif). Berhenti mencari flag yang hilang — carilah cara mengurangi byte/token atau menaikkan token/byte.

---

## 2. Koreksi terhadap `RESULTS.md`

`RESULTS.md` menyatakan: *"MTP GAGAL — GGUF Unsloth tidak punya MTP/nextn head... Untuk benar-benar dapat MTP: harus ganti ke GGUF ber-MTP, quant berbeda, download besar lagi."*

**Bagian pertama benar, kesimpulannya salah.** Yang benar:

- Model utama memang tidak punya tensor `nextn` (grep = 0 tensor — masih valid hari ini).
- Tapi MTP head Flash-Next adalah **satu blok qwen4exp utuh (4B param)** yang bisa berdiri sebagai **file draft terpisah** dan dipasang dengan `-md`. Bukan sesuatu yang harus menyatu dengan GGUF target.
- Unsloth sudah merilisnya di `unsloth/Qwen3.8-Flash-Next-GGUF` direktori `MTP/`:

  | File | Ukuran | Draft acceptance |
  |---|---|---|
  | `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2.60 GB | 66.1% |
  | `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` | 1.78 GB | 64.4% |
  | `mtp-Qwen3.8-Flash-Next-shared-BF16.gguf` | 4.87 GB | 66.5% |

- Ada juga jalur pihak ketiga `dzannotti/Qwen3.8-Flash-Next-MTP-GGUF` (2.5 GB Q4_K_M), eksplisit **"tested with unsloth's UD-Q4_K_XL and UD-IQ4_XS"** — persis file kita.

**Kenapa dulu gagal:** build kita `5d4a3be` (2026-08-31) belum punya graph MTP untuk `qwen4exp`. Dicek hari ini: di commit kita, case `LLM_ARCH_QWEN4EXP` punya **0** referensi nextn/mtp; di `origin/master` sekarang ada **35**. Errornya benar, tapi penyebabnya versi llama.cpp, bukan ketiadaan MTP head.

---

## 3. Yang kita ketinggalan di llama.cpp (40 commit)

Commit relevan antara `5d4a3be` dan `origin/master`:

| Commit | Isi | Dampak |
|---|---|---|
| `0eadefe` | `qwen4exp: support recurrent state rollback` (#28123) | **Prasyarat MTP** — rollback state SSM saat draft ditolak |
| `36b1015` | `qwen4exp: fix seq_cp, block position keying, cuda abort` (#27941) | Stabilitas |
| `09412af` | `qwen4exp: sum the indexer heads by slices` (#28023) | Perf indexer |
| `3466812` | `cuda: fuse MoE weighted expert reduction` (#25952) | **Decode MoE lebih cepat** |
| `e4b9af0` | `CUDA: XOR swizzle flash attn K,V smem fp16 tiles` (#25635) | Flash-attention lebih cepat |

Dua yang terakhir memberi speedup langsung tanpa MTP sama sekali `[estimate]` +3–10%.

---

## 4. Peringkat opsi

### ⭐ A+. UD-Q4_K_XL (existing) + MTP sidecar + mmap — **rekomendasi**

```bash
llama-server \
  -m models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75 \
  -ngl 999 -lm mmap -ot "per_layer_token_embd.weight=CPU" \
  -fa on --cache-type-k q8_0 --cache-type-v q8_0 -c 32768
```

- Download: **2.60 GB** saja. Model 104 GB yang sudah ada dipakai apa adanya.
- Retensi tetap **92.3%** — MTP itu *lossless*: draft yang salah ditolak, output identik dengan non-spec.
- Proyeksi **31–39 tok/s** `[estimate]`, dasar: Strix Halo (unified memory, kelas bandwidth mirip) 20.3 → 35.8 tok/s (+76%) di kode; M3 Max 27.4 → 38.8 (+42%). Prosa akan di ujung bawah, kode di ujung atas.
- `-lm mmap` menaruh PLE 26.82 GiB di **page cache** (reclaimable), bukan resident → resident ~**77 GiB**.

### A. UD-Q4_K_XL apa adanya (status quo)
21.8 tok/s, 92.3%, resident ~104 GiB. Titik aman yang sudah terbukti.

### B. blazux NVFP4 + vLLM
Resident ~76 GiB, decode 26 tok/s (NVFP4) / **31 tok/s** (hybrid fp8 side-layer), **prefill 1.500–2.000 tok/s** — 20–40× prefill kita. Stack docker + vLLM patched, bukan llama.cpp.
- Ini pemenang mutlak **kalau prompt panjang** (agentic, RAG, codebase). Prefill kita ~50–800 tok/s adalah kelemahan terbesar opsi A, dan MTP **tidak memperbaiki prefill sama sekali**.

### C. UD-IQ4_XS + MTP
93.7 GiB, retensi 89.6%, per-token ~5.3 GiB → baseline ~24 tok/s, dengan MTP ~34–43 `[estimate]`. Tukar 2.7 poin retensi untuk ~10% kecepatan. Kurang menarik dibanding A+.

### D. UD-IQ1_S
34.5 tok/s tapi 3.28 bpw — retensi jauh di bawah target 90%. Hanya untuk draft/eksperimen.

---

## 5. Bisa bikin recipe sendiri? Bisa — dan analisisnya menunjuk arah yang berlawanan dengan intuisi

Breakdown traffic per token, per family tensor (di luar experts & PLE):

| Tensor family | GiB | % traffic per token |
|---|---|---|
| `attn_qkv.weight` | 0.934 | 15.2% |
| `output.weight` | 0.629 | 10.2% |
| `attn_gate.weight` | 0.560 | 9.1% |
| `ssm_out.weight` | 0.560 | 9.1% |
| `attn_q.weight` | 0.374 | 6.1% |
| `ffn_gate_inp.weight` (router) | 0.234 | 3.8% |
| `attn_output.weight` | 0.187 | 3.0% |
| `hc_*` (hyper-connections) ×4 | 0.624 | 10.2% |

**Insight yang bisa dipakai:** aturan quant konvensional — "jaga attention tetap presisi tinggi, tekan experts" — **terbalik untuk kecepatan decode di model ini**:

- Experts = **69% file** tapi cuma **23% traffic/token**. Menekan experts Q4→Q3 memangkas file ~18 GiB tapi decode hanya +6% `[estimate]`, dengan biaya kualitas menyebar ke semua domain.
- attn/ssm = **3.7% file** tapi **63% traffic/token**. Menurunkan 4 family teratas (`attn_qkv`, `attn_gate`, `ssm_out`, `attn_q` = 2.43 GiB = 41% traffic) dari ~5 bpw ke 4 bpw memangkas ~0.49 GiB/token → **+9% decode** dengan file hampir tidak berubah.

Jadi recipe quant custom untuk hardware ini akan berbentuk: **experts dibiarkan setinggi mungkin (kualitas murah di sini), tensor per-token ditekan**. Ini kebalikan dari resep Baekpica yang menaikkan presisi edge layers.

**Tapi:** hasil maksimalnya cuma ~+10–15%, sementara MTP memberi +40–76% tanpa mengorbankan kualitas sama sekali. Dan bikin quant sendiri butuh source Q8_0 188 GB atau BF16 355 GB (download besar → wajib konfirmasi dulu) plus imatrix run berjam-jam.

**Kesimpulan: kerjakan MTP dulu. Custom quant baru masuk akal kalau setelah MTP masih kurang cepat**, dan lebih tepat diarahkan ke penghematan *memori* (biar muat berdampingan dengan Ollama) ketimbang kecepatan.

---

## 6. Rencana eksekusi bertahap

| # | Langkah | Biaya | Ekspektasi | Risiko |
|---|---|---|---|---|
| 1 | `git pull` llama.cpp + rebuild CUDA | ~15 mnt | +3–10%, unlock MTP | Rendah — build terpisah, biner lama disimpan |
| 2 | Re-bench baseline dengan build baru | ~10 mnt | Angka pembanding bersih | — |
| 3 | Tes `--spec-type ngram-mod` (**belum pernah dicoba** — kemarin kita pakai `ngram-cache`) | ~10 mnt | Gratis; sparkrun lapor sampai ~45 tok/s di kerja copy-heavy | — |
| 4 | Download MTP head Q8_0 (2.60 GB) | ~5 mnt | — | — |
| 5 | Bench MTP `n-max` 2 vs 3, prosa vs kode | ~20 mnt | **31–39 tok/s** | Sedang — spec bisa regresi di backend tertentu |
| 6 | Tambah `-lm mmap`, ukur resident | ~10 mnt | ~104 → ~77 GiB | Rendah |
| 7 | *(opsional)* blazux vLLM stack kalau prefill jadi masalah | ~2 jam + docker | prefill 20–40× | Tinggi — stack custom |

Total langkah 1–6: **± 1 jam**, download 2.6 GB. Di bawah ambang "tanya dulu" (>30 mnt proses / >10 GB download) kecuali langkah 7.

### Catatan memori — relevan dengan insiden 2026-09-01

`-lm mmap` bukan cuma soal hemat. PLE 26.82 GiB di page cache itu **reclaimable**: saat memori menipis, kernel membuang page cache dan model melambat. Tanpa mmap, alokasi itu resident dan menipisnya memori berujung `NVRM NV_ERR_NO_MEMORY` → stall → watchdog panic, persis yang terjadi 2026-09-01.

**`-lm mmap` mengubah mode gagal dari kernel panic menjadi sekadar lambat.** Itu alasan yang cukup untuk memakainya, terlepas dari angka resident.

Tetap begitu, resident ~77 GiB + KV masih belum nyaman berdampingan dengan beban Ollama sekarang (~36.5 GiB): 77 + 36.5 = 113.5 dari 121.7 GiB, dan page cache PLE akan terus diusir → decode anjlok. Decision #4 (stop Ollama sebelum serve) tetap berlaku.

## 7. Yang belum terverifikasi

- Angka speedup MTP Unsloth (1.67×, "83.2 → 138.8 tok/s") jelas **bukan** dari GB10 — hardware tidak disebut `[Belum Terverifikasi]`. Yang bisa dipindah ke kasus kita hanyalah **acceptance rate ~66%**.
- Proyeksi 31–39 tok/s diturunkan dari Strix Halo & M3 Max (sama-sama unified memory) `[estimate]`. PR #27836 mencatat CUDA unified memory dapat "gain serupa", tapi tidak ada angka GB10 spesifik yang saya temukan.
- `--spec-type ngram-mod` belum pernah kita ukur sama sekali.
- Retensi akurasi NVFP4 (opsi B): **diukur sendiri 2026-09-04**, setara dengan UD-Q4_K_XL
  pada GSM8K dan HumanEval+ — lihat bagian "Retensi akurasi: diukur" di bawah. Model card
  RadixArk melaporkan GSM8K 97.27 vs BF16 97.12–97.50 (vendor-reported).
- Tidak ada draft head DFlash/DSpark untuk Flash-Next (yang beredar hanya untuk Qwen3.8-27B) — jalur itu tertutup untuk sekarang.

## Sumber

- https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md
- https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF
- https://github.com/ggml-org/llama.cpp/pull/27836 · https://github.com/ggml-org/llama.cpp/discussions/27864
- https://github.com/styles01/sparkrun-recipes/blob/main/runbooks/qwen38-flash-next-image.md
- https://blog.kubesimplify.com/running-qwen3-8-flash-next-on-dgx-spark-and-rtx-pro-6000
- https://github.com/blazux/qwen3.8-Flash-DGX
- https://forums.developer.nvidia.com/t/qwen3-8-flash-next-ud-q4-k-xl-gguf-on-dgx-spark-with-llama-cpp-gpu-experts-ple-n-gram-table-streamed-from-disk-25-tok-s-up-to-1m-context/381720
- https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/


---

# Hasil eksekusi 2026-09-03

Semua angka dari `runs/bench-2026-09-03.jsonl`, 3 run per konfigurasi, `n_predict` 256, temperature 0.

## Temuan metodologis: benchmark lama melebih-lebihkan 14.6%

`bench.sh` versi lama mengirim **prompt yang sama 3 kali dengan temperature 0**. Karena
output-nya identik tiap run, server bisa memanfaatkan slot/KV reuse — dan untuk
speculative decoding, n-gram cache tinggal menyalin output run sebelumnya.

| Metode | build lama | build baru | + ngram-mod |
|---|---|---|---|
| A: prompt sama diulang 3× | 21.80 | **27.18** | **86.05** |
| B: 5 prompt berbeda dirotasi | 23.62 | **23.71** | **23.58** |

Selisih A vs B pada build yang sama: **+14.6%** tanpa spekulasi, dan **+263%** dengan
ngram-mod. `bench.sh` sekarang merotasi 5 prompt berbeda; metode A tidak boleh dipakai lagi.

## Hasil sebenarnya (metode B)

| Konfigurasi | Median tok/s | vs baseline | Draft accepted |
|---|---|---|---|
| build 5d4a3be (lama) | 23.62 | — | — |
| build 0ba6499 (63 commit lebih baru) | 23.71 | **+0.4%** | — |
| + `--spec-type ngram-mod` | 23.58 | **-0.6%** | 0 |
| + `--spec-type draft-mtp` | — | **GAGAL LOAD** | — |

**Update llama.cpp tidak memberi percepatan decode yang terukur.** `cuda: fuse MoE
weighted expert reduction` (#25952) dan `XOR swizzle flash attn` (#25635) tidak
mengubah apa pun untuk model ini pada beban single-stream.

**ngram-mod tidak memberi gain di prosa** — nol draft diterima dengan prompt bervariasi,
konsisten dengan hasil `ngram-cache` 2026-09-01 (acceptance 16%). Kesimpulan lama bahwa
ngram "belum pernah dicoba dan mungkin berbeda" sudah terjawab: sama saja.

## MTP terblokir di upstream llama.cpp

Kedua varian head Unsloth ditolak build 0ba6499:

| Varian | Ukuran | Tensor | Error |
|---|---|---|---|
| `mtp-...-shared-Q8_0.gguf` | 2.60 GiB | 32 | `tensor 'token_embd.weight' not found` |
| `mtp-...-Q8_0.gguf` (standalone) | 3.85 GiB | 34 | `tensor 'output_hc_norm.weight' not found` |

Diverifikasi dengan membandingkan set tensor terhadap model target (1224 tensor):

| Tensor | target | MTP standalone | MTP shared |
|---|---|---|---|
| `token_embd.weight` | ada | ada | **tidak** |
| `output.weight` | ada | ada | **tidak** |
| `output_hc_norm.weight` | ada | **tidak** | **tidak** |

Upstream memperlakukan draft model sebagai model `qwen4exp` utuh dan mewajibkan set
tensor lengkap (`LLM_TENSOR_HC_HEAD_NORM` terdaftar di `llama-arch.cpp:522`). Head
Unsloth dibuat untuk fork `unslothai/llama.cpp` yang tahu ini head parsial.

**Jalan yang tersisa untuk MTP:**
1. Build fork `unslothai/llama.cpp` tag `b10715-mix-86bd2d3` atau lebih baru.
2. Tunggu PR #27836 merge — masih berstatus draft, plus butuh commit detached-head
   loader `crusaderky` a82a58a.
3. Tambahkan `output_hc_norm.weight` ke GGUF head sendiri (patch tensor) — belum diuji,
   dan belum tentu cukup karena mungkin ada tensor wajib lain.

## Status memori (terukur, bukan estimasi)

Dengan default `--load-mode auto` (yang ternyata sudah mmap):

- used saat model ter-load: **83.4-84.3 GiB**, bukan ~104 GiB seperti dugaan awal
- page cache: ~24 GiB
- available tersisa: ~35 GiB
- load time: **44.5 detik** (warm)

Klaim bahwa `-lm mmap` akan menurunkan resident 104 → 77 GiB **tidak relevan** — default
sudah mmap. Manfaat tambahan flag itu belum terukur.

## Kesimpulan yang bertahan

Yang masih berdiri dari analisis awal: **decode dibatasi memory bandwidth**. Efisiensi
jujur 57% dari roofline 41.4 tok/s. Tapi tidak ada jalur murah untuk menaikkannya —
update build nol, ngram nol, MTP terblokir. Yang tersisa nyata:

1. **Fork unslothai** untuk MTP — satu-satunya jalur yang punya bukti angka di hardware
   sekelas (Strix Halo 20.3 → 35.8).
2. **blazux NVFP4 + vLLM** — 31 tok/s hybrid, prefill 20-40× lebih cepat. Stack berbeda.
3. **Terima 23.7 tok/s** dengan retensi 92.3%.

---

# Hasil fork Unsloth + beban kerja coding agent (2026-09-03, sesi lanjutan)

Workload sebenarnya: **coding agent dengan prompt panjang**. `bench.sh` (prosa, prompt ~30
token) tidak representatif untuk itu, jadi dibuat `bench-code.sh`: prompt ~10.9K token dari
source `llama.cpp` nyata + task coding, 3 file & 3 task berbeda per run.

Fork Unsloth dipakai lewat **prebuilt** `app-b10715-mix-86bd2d3-linux-arm64-cuda13-portable`
(180 MB, tanpa compile) — `BUILD=fork ./serve.sh`.

## MTP BERHASIL di fork

Prompt 10.886 token, output 256 token, median dari 3 run:

| Konfigurasi | Prefill tok/s | TTFT | Decode tok/s | Acceptance | RAM |
|---|---|---|---|---|---|
| upstream `0ba6499` | 337.8 | 33.1s | 22.11 | — | 84.3 GiB |
| fork `b10715` | **525.3** | **20.8s** | 24.07 | — | 84.3 GiB |
| fork + MTP n-max 2 | 505.6 | 21.7s | **36.45** | **94.3%** | 88.8 GiB |
| fork + MTP n-max 3 | 505.7 | 21.7s | **37.81** | 92.8% | 89.1 GiB |

**Dua kemenangan terpisah:**

1. **MTP: decode 24.07 → 37.81 tok/s (+57%)**, acceptance **92–94%** di kode. Ini jauh di atas
   ekspektasi dari data Strix Halo (0.90 di kode) dan menjelaskan kenapa uji prosa kemarin
   menyesatkan — prosa adalah kasus terburuk untuk MTP, kode kasus terbaik. Lossless: draft
   yang salah ditolak, output identik dengan non-spec.
2. **Fork lebih cepat prefill: 337.8 → 525.3 tok/s (+55%)**, TTFT 33.1s → 20.8s, bahkan tanpa
   MTP. Ini tidak terduga dan bukan efek MTP.

Biaya MTP: **+4.5 GiB** RAM (84.3 → 88.8 GiB) untuk head Q8_0.

## End-to-end untuk satu panggilan agent

| Konfigurasi | TTFT | Decode | Total | Porsi prefill |
|---|---|---|---|---|
| upstream | 33.1s | 11.6s | **44.7s** | 74% |
| fork | 20.8s | 10.6s | 31.5s | 66% |
| fork + MTP n3 | 21.7s | 6.8s | **28.5s** | **76%** |

**Perbaikan end-to-end 44.7s → 28.5s (+57%).** Konfigurasi yang direkomendasikan:
`BUILD=fork ./serve.sh --mtp --nmax 3`.

## Tapi prefill tetap masalah dominan

Setelah MTP, **76% waktu per panggilan habis di prefill**. Decode sudah bukan bottleneck.
Untuk coding agent, ini yang menentukan pengalaman pakai:

- prompt 10.9K → TTFT ~21s
- prompt 32K → TTFT ~63s `[estimate]`, ekstrapolasi linear dari 505 tok/s

blazux NVFP4+vLLM mengklaim prefill **1.500–2.000 tok/s** (3–4× fork ini) plus prefix caching
dengan TTFT 1.4s untuk 20K token yang sudah pernah dilihat. Untuk agent yang memanggil
berulang dengan konteks yang tumbuh, prefix caching itu bisa lebih besar dampaknya daripada
angka prefill mentah.

**Rekomendasi bertingkat:**
1. **Sekarang:** pakai `BUILD=fork ./serve.sh --mtp --nmax 3`. Gratis, sudah terukur, +57%.
2. **Berikutnya:** evaluasi blazux vLLM khusus untuk prefill + prefix caching. Butuh download
   bobot NVFP4 ~122 GiB (konfirmasi dulu) dan retensi akurasi NVFP4 belum terukur pihak ketiga.
3. Konsekuensi memakai fork: tertinggal dari upstream, bergantung rilis Unsloth. Untuk sekarang
   sepadan — upstream tidak bisa memuat MTP head sama sekali.

---

# Perbandingan final: llama.cpp fork+MTP vs vLLM NVFP4 (2026-09-03 malam)

Diukur dari sisi klien lewat streaming SSE untuk kedua stack (`bench-stream.py`), prompt
identik ~10.9K token dari source nyata, 256 token output, `finish_reason=length` di semua
run. Jumlah token diambil dari `usage.completion_tokens` yang dilaporkan server.

## Hasil

| Stack | TTFT | Decode | Total/panggilan | RAM | Load |
|---|---|---|---|---|---|
| llama.cpp fork+MTP n3 | 21.59s | **37.77** | 28.4s | **88 GiB** | **45s** |
| vLLM NVFP4 | **11.30s** | 30.27 | **19.8s** | 112 GiB | 14 mnt |
| llama.cpp — prefix cache hit | 1.82s | 35.83 | **9.0s** | | |
| vLLM — prefix cache hit | 1.86s | 30.71 | 10.2s | | |

- **Cold** (konteks baru tiap panggilan): vLLM **1.44× lebih cepat** — prefill 1.9× menutupi
  decode yang 1.25× lebih lambat.
- **Warm** (prefix sama): llama.cpp **1.14× lebih cepat**, dan **prefix caching keduanya
  praktis setara (1.82s vs 1.86s)**.
- **Titik impas: vLLM menang bila >12% panggilan bersifat cold.**

## Koreksi: prefix caching BUKAN keunggulan khas vLLM

Sepanjang riset ini aku memperlakukan prefix caching sebagai alasan utama pindah ke vLLM,
mengutip klaim blazux "TTFT 1.4s untuk 20K token". Terukur: **llama.cpp punya kemampuan yang
sama dan sama cepatnya** (1.82s vs 1.86s pada prompt 10.9K). Argumen pindah stack karena
prefix caching tidak berdiri.

## Tiga cacat metodologi yang ditemukan hari ini

Semuanya menghasilkan angka yang tampak masuk akal padahal mengukur hal lain:

1. **Prompt berulang + temperature 0** → inflasi 14.6% tanpa spekulasi, 263% dengan ngram-mod.
2. **`"model": "x"`** → vLLM balas HTTP 404 (llama.cpp mengabaikan field ini). Nama model kini
   dibaca dari `/v1/models`.
3. **Menghitung chunk SSE sebagai token** → vLLM mengemas ~2.6 token per chunk, membuat decode
   tampak 9.7 tok/s padahal 30.3. Kini memakai `usage.completion_tokens`.

Ditambah: request pertama setelah server start kadang gagal/kosong dan ikut masuk median.
`bench-stream.py` sekarang membuang satu run warm-up dan menolak run gagal secara eksplisit.

## Rekomendasi: tetap di llama.cpp fork + MTP

```bash
BUILD=fork ./serve.sh --mtp --nmax 3
```

vLLM unggul 1.44× hanya pada panggilan cold. Untuk coding agent dengan system prompt stabil
dan konteks yang bertumbuh di ujung, mayoritas panggilan adalah cache hit — wilayah di mana
llama.cpp justru sedikit lebih cepat. Di luar kecepatan:

| | llama.cpp fork | vLLM NVFP4 |
|---|---|---|
| RAM | 88 GiB (sisa 33) | 112 GiB (sisa ~8) |
| Restart | 45 detik | 14 menit |
| Retensi akurasi (diukur, thinking off) | GSM8K 97.3%, HumanEval+ 93.9% | GSM8K 97.0%, HumanEval+ 95.7% |
| Stack | satu binary | docker + vLLM patched |

Margin memori vLLM (~8 GiB) berbahaya di mesin ini mengingat insiden watchdog 2026-09-01.

**Pindah ke vLLM baru masuk akal kalau** beban kerjanya ternyata didominasi konteks yang
selalu baru (>12% cold), misalnya memindai banyak file berbeda tiap panggilan. Itu bisa
diukur dari pola pemakaian nyata, bukan ditebak sekarang: `./cache-ratio.py -v` membaca
`runs/serve-current.log` setelah sesi nyata dan melaporkan rasio cold call terhadap titik
impas ini. Log benchmark sengaja 100% cold (prompt berbeda tiap run), jadi tidak mewakili.
**Sudah diukur 2026-09-05** (bagian paling bawah): 8.7% cold, 95.9% token dari cache.

---

## 2026-09-04: Crash CUDA saat prefill — fork DAN upstream, mitigasi `-ub 256`

Ditemukan saat menjalankan `bench-accuracy.py` (HumanEval+): llama-server mati di soal
`HumanEval/68` dengan

```
CUDA error: an internal operation failed
  in function ggml_cuda_mul_mat_cublas_impl at ggml-cuda.cu:1623 (cublasGemmEx)
```

Reproduksi deterministik: server baru start, satu request prompt itu (381 token prompt,
thinking off), mati dalam 1 detik. Bukan OOM — available 116 GiB saat itu.

**Isolasi** (`runs/crash-sweep-2026-09-04.jsonl`, log crash di `runs/serve-crash-*.log`):

| Konfigurasi | Prompt HumanEval/68 |
|---|---|
| fork Unsloth b10715 + MTP | crash |
| fork Unsloth b10715 tanpa MTP | crash |
| upstream `0ba6499` (build-new) `--ngram-mod` | crash |
| fork + MTP, **`-ub 256`** | **jalan** (474 token completion) |

Jadi bukan MTP dan bukan fork — bug backend CUDA llama.cpp di GB10 (sm_121), ada di
upstream juga. Sweep prompt netral (token id mentah lewat `/completion`, `cache_prompt`
off) ukuran 300–600 pada `-ub` default: hanya **367 dan 512** yang crash; tapi 381 netral
*tidak* crash padahal prompt HumanEval/68 (381 token) crash. Pemicunya **bergantung data,
bukan sekadar ukuran batch** — `[Inferensi]` bentuk GEMM per-expert (berapa token yang
dirutekan ke tiap expert dalam satu ubatch) yang menabrak jalur cuBLAS bermasalah.

**Mitigasi:** `-ub 256`. Dengan itu sweep 1–600 bersih dan HumanEval/68 jalan. Sekarang
default di `stack.sh` (`UBATCH=256`, bisa di-override). Ini menghindari semua kasus yang
bisa kami reproduksi, bukan jaminan — akar masalahnya belum diperbaiki di upstream.
Kenapa ini tidak pernah muncul di benchmark 2026-09-03: prompt ~10.9K token diproses
dalam ubatch 2048/512 penuh; hanya ubatch terakhir yang berukuran acak, dan tiga prompt
tidak cukup untuk menabraknya. Coding agent yang mengirim ratusan prompt beragam **akan**
menabraknya cepat atau lambat — 468 request eval menabraknya di request ke-369.

Dilaporkan ke upstream sebagai **ggml-org/llama.cpp#28377** (2026-09-04) dengan reproduksi
sweep di atas. Kandidat terkait: #28251 (call site sama, `cublasGemmEx` di jalur MoE, status
cuBLAS berbeda, RTX 3070) dan #27792 (OOB read di MMQ `mul_mat_id`, jalur kernel berbeda
tapi sama-sama bergantung ubatch) `[Spekulasi]`.

**Biaya `-ub 256`** (`runs/bench-stream2.jsonl`, label `llamacpp-mtp-ub256` vs `-ub512`,
prompt ~10.9K token, median 2 run valid): TTFT **26.2 s vs 21.9 s** (+20% prefill),
decode 36.7 vs 37.2 tok/s (setara). Prefill memang menjadi lebih mahal; itu harga
stabilitas sampai upstream memperbaiki kernelnya. Catatan: run ke-4 (sumber
`speculative.cpp`) gagal di kedua konfigurasi — server mengembalikan 1 token lalu stop.
Prompt yang sama jalan normal di vLLM pada 2026-09-03. Belum diselidiki `[Belum Terverifikasi]`.

---

## 2026-09-04: Retensi akurasi — diukur, bukan dari model card

Pertanyaan yang tersisa dari perbandingan backend: apakah NVFP4 (RadixArk) lebih buruk dari
UD-Q4_K_XL (Unsloth)? BF16 360 GB tidak muat di GB10, jadi yang diukur adalah **kedua quant
pada task set, harness, dan setting decoding yang identik**, lalu dibandingkan satu sama lain
dan dengan angka BF16 yang dipublikasikan. Alat: `bench-accuracy.py`; data:
`runs/accuracy-2026-09-04.jsonl`; sampel per soal di `tmp/accuracy/<label>/` (tidak di-track).

Setting: thinking **off** (`chat_template_kwargs.enable_thinking=false`, kedua server
menghormatinya — 27 prompt token identik), temperature 0, seed 1234. GSM8K: lm-eval
`gsm8k_cot_zeroshot`, 300 soal test pertama, max 1024 token. HumanEval+: evalplus, 164 soal,
greedy, max 2048 token (default 768 memotong dua jawaban yang menalar panjang dulu).

| | llama.cpp UD-Q4_K_XL + MTP | vLLM NVFP4 (RadixArk) | BF16 published |
|---|---|---|---|
| GSM8K (300, final-paragraph) | **97.3%** (292) | **97.0%** (291) | 97.1–97.5 (RadixArk card, thinking on) |
| GSM8K (300, lm-eval flexible-extract) | 86.3% | 85.3% | — |
| HumanEval+ pass@1 base | 96.3% (158/164) | 97.0% (159/164) | — |
| HumanEval+ pass@1 plus | **93.9%** (154/164) | **95.7%** (157/164) | — |

stderr GSM8K ≈ 2.0 poin; HumanEval+ ≈ 1.9 poin. **Kesimpulan: kedua quant setara dalam
batas noise.** Bukti yang lebih kuat dari angka agregatnya adalah *soal mana* yang gagal:

- GSM8K: 7 soal gagal di keduanya; llama.cpp gagal 1 soal tambahan, vLLM 2. Sisa kegagalan
  adalah batas model (mis. 36.36 vs 36, pembulatan), bukan efek quant.
- HumanEval+: 7 soal gagal di keduanya; **set kegagalan vLLM adalah subset ketat** dari
  llama.cpp, yang gagal 3 soal tambahan (38, 116, 124). Selisih 1.8 poin, di dalam stderr,
  tapi arahnya konsisten: NVFP4 tidak lebih buruk, kalau ada malah sedikit lebih baik.

Baris tabel perbandingan backend di atas ("belum terukur") sudah dikoreksi. Angka 92.3%
Unsloth tidak sebanding dengan ini (metrik "top-1% accuracy" vs BF16, bukan task score).

**Dua cacat harness yang ditemukan dan diperbaiki** (lanjutan daftar cacat 2026-09-03):

4. lm-eval `flexible-extract` mengambil angka *terakhir* di respons. Model ini menutup dengan
   "Kylar needs to pay **$64** for the 16 glasses" → terbaca 16. Juga `$26.00` ≠ `26`. Dari 48
   "kegagalan" versi lm-eval, 37 adalah jawaban benar. Scorer `acc_final_para` (angka
   terakhir di paragraf terakhir, angka tebal diutamakan) memperbaikinya tanpa satu pun
   verdict benar berubah jadi salah. Kedua angka disimpan; baca yang final-paragraph.
5. llama.cpp + MTP dengan 4 slot paralel: 96.3% vs 97.3% dengan 1 slot, dan satu respons
   berhenti setelah 5 karakter ("Let $"). Cocok dengan gejala upstream #28286 (kontaminasi
   antar slot dengan draft-mtp). Hasil resmi memakai 1 slot; vLLM boleh 4 paralel.

---

## 2026-09-05: Rasio cache hit nyata — sesi coding agent pertama

Angka yang sebelumnya kosong di bagian Rekomendasi. Sumber: `runs/cache-ratio-2026-09-05.jsonl`
(23 request, direkonstruksi dari log llama-server oleh `cache-ratio.py`; kolom
`draft_acceptance` ditambahkan dari baris `print_timing` yang sama). Beban: dua sesi Pi dari
laptop lewat port-forward NVIDIA Sync — satu one-shot "mini spreadsheet" (satu file HTML,
parser formula + dependency graph) lalu satu sesi follow-up (fix bug + fitur baru) di file yang
sama. Model menulis kode, menjalankan test Node, memperbaiki test-nya sendiri; total 23 request,
konteks tumbuh dari 1.6k ke 40k token.

| | Nilai |
|---|---|
| Prompt tokens total / cached | 606 854 / 582 177 (**95.9%**) |
| Cold call (cached < 50%) | **2 / 23 = 8.7%** — di bawah titik impas 12% |
| Prefill wall total | 66.0 s, 2.87 s per call; **37.1 s** di antaranya satu call |
| Prefill per call kalau call itu dikecualikan | ~1.3 s (22 call, 29 s) |
| Completion tokens total / decode wall | 35 461 / 1076 s → **33.0 tok/s** agregat |
| Decode per call | 26.6–38.5 tok/s; ≥30k konteks cenderung 27–30 |
| Draft acceptance MTP agregat | 0.856 (20 670 / 24 145), per call 0.78–0.99 |

**Kesimpulan:** pola pemakaian coding agent sungguhan jatuh di wilayah llama.cpp + MTP, seperti
diasumsikan Rekomendasi. Sampel kecil (satu pengguna, dua sesi, satu jenis tugas), tapi arahnya
jelas: 21 dari 23 call cache hit ≥ 82%, dan 17 di antaranya ≥ 98%.

**Dua cold call itu bukan yang diduga.** Yang pertama adalah probe awal (wajar). Yang kedua,
task 6875, adalah **turn kedua** sesi pertama: prompt 18 364 token, hanya 1 988 cached, 16 376
token di-prefill ulang selama 37 s. Ukuran yang di-prefill ulang hampir persis sama dengan
respons sebelumnya (16 342 token, 456 s decode — respons one-shot berisi thinking panjang plus
seluruh file). `[Inferensi]` Isi respons itu dikirim balik oleh Pi sebagai history, tapi
prefix cache tidak match: kandidat penyebab (a) token hasil generate ≠ hasil re-tokenize teks
yang sama, sehingga satu token beda di awal menggugurkan semua setelahnya, atau (b) chat
template me-render ulang blok thinking dengan format yang berbeda dari yang di-generate. Mana
yang benar belum diuji; setelah turn itu, semua turn berikutnya hit ≥ 90%, jadi cache-nya
bekerja normal — miss ini hanya terjadi sekali per percakapan, tepat setelah respons pertama.
Kalau pola ini konsisten, biaya sesi = satu prefill sebesar respons pertama; untuk respons
pendek biayanya kecil, untuk one-shot 16k token seperti ini 37 s.

**Dipersempit hari yang sama.** Dua sumber: capture traffic Pi di laptop (socat) dan
`cache-probe.py` di server.

*Client (capture socat, percakapan Pi 2 turn tanpa tool):* Pi **mengirim balik `reasoning_content`**
(28 kemunculan di body request), dan turn 2-nya hit 99.0% (task 15198). Jadi hipotesis "Pi membuang
thinking" GUGUR untuk turn teks biasa.

*Server (`cache-probe.py`, `usage.prompt_tokens_details.cached_tokens` dibaca langsung, turn 2
`max_tokens=1`):* setiap bentuk turn yang dikirim balik apa adanya **HIT**:

| Bentuk turn 1 | non-streaming | streaming (delta dirakit, arguments di-serialise ulang) |
|---|---|---|
| thinking + teks | HIT | HIT |
| thinking + tool call | HIT | HIT |
| thinking + teks + tool call | HIT | HIT |
| ... dengan tool call 3.6k token (file HTML) | HIT | HIT |
| thinking panjang (puzzle) + tool call | — | HIT |

Dan diff token history yang di-render ulang vs token yang di-generate: **identik** untuk semua bentuk.
Yang memecah cache hanya tiga hal (diff deterministik via `/apply-template`): `reasoning_content`
dibuang atau dipindah ke field lain (`reasoning` diabaikan template); urutan key `arguments` berubah;
newline di akhir nilai parameter di-strip. Source Pi (`packages/ai/src/api/openai-completions.ts`)
tidak melakukan satu pun dari ketiganya: field mengikuti signature yang diterima (`reasoning_content`),
`arguments` = `JSON.stringify` dari object hasil parse (urutan key terjaga), `content` kosong dibuang
(render identik).

**Mengapa satu mismatch berharga seluruh respons.** Model ini `qwen4exp`: layer SSM
(`ssm.state_size` 128, `full_attention_interval` 4 → 3 dari 4 layer linear attention). State
recurrent tidak bisa dimundurkan ke posisi sembarang, jadi llama-server hanya bisa kembali ke
**context checkpoint** (`--ctx-checkpoints` 32, `--checkpoint-min-step` 8192; dibuat saat prompt
processing, bukan saat generate). Mismatch di mana pun dalam turn → rollback ke checkpoint akhir prompt
turn sebelumnya → prefill ulang seluruh respons. Terlihat di probe: variant tanpa reasoning selalu
`cached` = ukuran prompt turn 1 (atau checkpoint lebih awal), bukan posisi mismatch. Pola task 6875
(`cached` 1988 ≈ prompt turn 1 sebesar 1992) adalah tanda tangan rollback ini; mismatch-nya sendiri
bisa di mana saja dalam 16k token itu.

**Status: belum tereproduksi dari sisi server.** Reproduksi paling setia (prompt spreadsheet asli via
`write_file`, streaming) menabrak `max_tokens` 24000 tanpa selesai (682 s decode) sehingga tidak
konklusif. Langkah penentu berikutnya dieksekusi saat sesi Pi nyata: server sendiri bisa mencetak
token di sekitar mismatch. Di launch script Sync:

```bash
LLAMA_SERVER_SLOTS_DEBUG=1 LLAMA_SERVER_SLOTS_N_DIFF=12 API_KEY=<key> ./stack.sh start llamacpp
```

lalu ulangi tugas Pi yang berakhir dengan tool call besar, dan cari di `runs/serve-current.log`
baris `old: ... | ...` / `new: ... | ...` (WARN, tampil tanpa verbose) — token sebelum `|` cocok,
sesudahnya adalah mismatch-nya. Env var diwariskan `stack.sh` → `serve.sh` → `llama-server`.

Catatan `cache-ratio.py`: verdict "vLLM territory" 18.5% setelah sesi ini terpolusi probe sintetis
(3 dari 5 cold call adalah probe). Tanpa probe: 3/25 = 12.0%, dan dua di antaranya adalah turn
pertama percakapan baru yang memang selalu cold. Rasio cold dengan demikian bergantung pada panjang
percakapan, bukan hanya pola tugas.

**Catatan decode:** 33 tok/s agregat vs 36.7 tok/s di benchmark. Selisihnya konsisten dengan
konteks yang jauh lebih panjang (benchmark ~2k, sesi ini sampai 40k) dan acceptance MTP yang
lebih rendah di kode yang belum pernah ada (0.78–0.89) dibanding saat menyalin ulang (0.99).
Angka benchmark tidak salah; angka ini yang mewakili pemakaian nyata.
