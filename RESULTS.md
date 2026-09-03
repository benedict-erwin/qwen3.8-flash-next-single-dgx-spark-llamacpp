# Hasil benchmark opsi A di DGX Spark GB10 (2026-09-01)

> **KOREKSI 2026-09-02 — baca `OPTIMIZATION.md`.** Kesimpulan "MTP tidak on the table"
> di bawah **tidak berlaku lagi**. MTP head Flash-Next tersedia sebagai file sidecar
> terpisah (Unsloth `MTP/mtp-*-shared-Q8_0.gguf`, 2.60 GB) yang dipasang dengan `-md`
> di samping UD-Q4_K_XL yang sudah ada — tidak perlu ganti GGUF atau download ulang.
> Kegagalan `--spec-type draft-mtp` kemarin disebabkan build llama.cpp `5d4a3be` belum
> menyambungkan graph MTP ke arsitektur `qwen4exp`; `origin/master` sekarang sudah.

Model: **unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL** (4 part, 104 GiB total, retensi akurasi 92.3% per Unsloth).
Runtime: llama.cpp master `5d4a3be` (qwen4exp), CUDA aarch64.
Config: `-ngl 999`, PLE n-gram → CPU (`-ot per_layer_token_embd.weight=CPU`), flash attention, **KV cache q8_0**, ctx 32768.
Env: memory 113 GiB free saat test (Ollama idle). Model hangat di page-cache → load < 45s (bukan ~1.5 min cold).

## Decode throughput (256 token greedy, 3 run)

| Config | Median decode | Runs | Catatan |
|---|---|---|---|
| baseline (KV q8_0, no spec) | **21.8 tok/s** | 21.8 / 21.8 / 24.4 | sesuai rentang publik 22–27 tok/s |
| + ngram self-speculative | 24.3 tok/s | 21.9 / 24.3 / 24.7 | dalam noise; draft acceptance 6/38 (~16%) di prosa → net ≈ 0 |

Prefill di prompt pendek noisy (9–54 tok/s) karena caching — tidak representatif; abaikan.

## Temuan kunci: MTP TIDAK tersedia di GGUF ini

`--spec-type draft-mtp` **gagal load**: llama.cpp `context type MTP requested but model doesn't contain MTP layers`. GGUF Unsloth UD-Q4_K_XL **tidak menyertakan MTP/nextn head** (grep `nextn` = 0 tensor).

→ Angka "+21% MTP" di FINDINGS.md berasal dari build **Baekpica** yang meng-embed MTP sendiri, BUKAN dari GGUF Unsloth. Asumsi awal bahwa MTP head ikut di quant Unsloth = **salah, terkoreksi**.

Alternatif tanpa MTP tensors:
- **ngram self-speculative** (dicoba): net ≈ 0 di prosa. Mungkin membantu di output kode/terstruktur (acceptance lebih tinggi) — belum diuji.
- Untuk benar-benar dapat MTP: harus ganti ke GGUF ber-MTP (`dzannotti/Qwen3.8-Flash-Next-MTP-GGUF` atau Baekpica) — quant berbeda, kualitas belum terukur, dan download besar lagi di link ter-throttle ~11 MB/s.

## Kesimpulan

Opsi A apa adanya = **~22 tok/s, retensi 92.3%**, setup Ollama/llama.cpp standar, MTP tidak on the table dengan file ini. Ini titik "akurasi terjamin" yang solid. Untuk mengejar decode lebih tinggi (+MTP), keputusannya ada di FINDINGS.md opsi B/C — perlu download & stack berbeda.
