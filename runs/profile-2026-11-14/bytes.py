#!/usr/bin/env python3
"""Per-launch byte accounting for an nsys decode capture: bytes.py <capture.sqlite> [t0_frac]
Maps every weight-reading GEMV launch to the matrix it reads (type + rows from the launch grid,
columns from the GGUF inventory) and reports effective GB/s in situ, kernel busy time and idle time."""
import sqlite3, re, sys, collections
MiB = 2**20; BUS = 273e9
c = sqlite3.connect(sys.argv[1])
rows = c.execute("""select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockY from CUPTI_ACTIVITY_KIND_KERNEL k
                    join StringIds s on s.id=k.demangledName order by k.start""").fetchall()
# bytes per row-of-matrix: bits per weight incl. block overhead
BPW = {8: 8.5, 12: 4.5, 13: 5.5, 7: 6.0, 2: 4.5, 20: 4.5}   # Q8_0, Q4_K, Q5_K, Q5_1
def mat_bytes(ncols, nrows, t): return ncols * nrows * BPW[t] / 8
fam = collections.defaultdict(lambda: [0, 0.0, 0.0])   # n, time_s, bytes
unk = collections.Counter()
ivs = []; n_target = n_draft = 0; tok_cols = collections.Counter()
for st, en, name, gx, gy, gz, by in rows:
    d = (en - st) / 1e9; ivs.append((st, en))
    m = re.match(r'void (mul_mat_vec_q(?:_moe)?)<\(ggml_type\)(\d+), \(int\)(\d+), \(bool\)(\d)(?:, \(bool\)(\d), \(bool\)(\d))?>', name)
    if not m:
        k = re.sub(r'<.*', '', name.replace('void ', '')).split('(')[0]
        f = ('other: ' + ('elementwise' if re.search(r'k_bin_bcast|unary|rms_norm|scale|concat|silu|sigmoid|k_sum|k_copy|cpy|clamp|k_repeat|soft_max|argmax|top_k|hc_', k) else k))[:34]
        fam[f][0] += 1; fam[f][1] += d; continue
    kern, t, a, b1, b2, b3 = m[1], int(m[2]), int(m[3]), m[4] == '1', m[5] == '1', m[6] == '1'
    if kern == 'mul_mat_vec_q_moe':                      # <type, rows_per_block, fusion>; blockY = tokens, gridY = experts
        nrows = gx * 2; fused = b1; ntok = by
        if nrows == 640:  ncols, lab = 2560, 'MoE gate+up' if fused else 'MoE gate/up'
        elif nrows == 2560: ncols, lab = 640, 'MoE down'
        else: unk[(kern, t, nrows)] += 1; continue
        byt = gy * ntok * mat_bytes(ncols, nrows, t) * (2 if fused else 1)
        f = '%s t%d (%d tok)' % (lab, t, ntok)
    else:                                                # <type, ncols_dst, fusion, small_k, halve>
        ncd, fused, small_k = a, b1, b2
        rpb = 2 if ncd > 1 else (by if small_k else 1)
        nrows = gx * rpb; us = d * 1e6
        if gy > 1 and nrows in (640, 2560):              # single-token expert GEMV
            ncols = 2560 if nrows == 640 else 640
            byt = gy * mat_bytes(ncols, nrows, t) * (2 if fused else 1)
            f = 'MoE %s t%d (1 tok)' % ('gate+up' if nrows == 640 else 'down', t)
        else:
            if nrows == 248320: ncols, lab = 2560, ('lm_head target' if t == 8 else 'lm_head draft')
            elif nrows == 12288: ncols, lab = 2560, 'attn_q'
            elif nrows == 6144: ncols, lab = 2560, 'attn_gate'
            elif nrows == 5120: ncols, lab = 2560, 'eh_proj'
            elif nrows == 10240: ncols, lab = (2560, 'attn_qkv/ple_key') if us > 80 else (320, 'hc_up')
            elif nrows == 2560: ncols, lab = (6144, 'ssm_out/attn_output') if us > 40 else (640, 'shexp_down')
            elif nrows == 640: ncols, lab = 2560, 'shexp gate/up'
            elif nrows == 512: ncols, lab = 2560, 'attn_k/v'
            elif nrows == 320: ncols, lab = 10240, 'hc_down'
            else: unk[(kern, t, nrows, gy)] += 1; fam['other: small gemv'][0] += 1; fam['other: small gemv'][1] += d; continue
            byt = gy * mat_bytes(ncols, nrows, t) * (2 if fused else 1)
            if lab == 'lm_head target': n_target += 1; tok_cols[ncd] += 1
            if lab == 'lm_head draft': n_draft += 1
            f = 'dense %s' % lab
    fam[f][0] += 1; fam[f][1] += d; fam[f][2] += byt
# busy union
ivs.sort(); busy = 0; cs, ce = ivs[0]
for s, e in ivs[1:]:
    if s > ce: busy += ce - cs; cs, ce = s, e
    else: ce = max(ce, e)
busy += ce - cs; span = ivs[-1][1] - ivs[0][0]
print('span %.3f s, GPU busy %.3f s (%.1f%%), idle %.3f s; target forwards %d (verify widths %s), draft forwards %d'
      % (span/1e9, busy/1e9, 100*busy/span, (span-busy)/1e9, n_target, dict(sorted(tok_cols.items())), n_draft))
tw = sum(v[1] for k, v in fam.items() if v[2]); bw = sum(v[2] for v in fam.values()); to = sum(v[1] for k, v in fam.items() if not v[2])
print('%-34s %7s %9s %9s %8s %8s' % ('family', 'n', 'time ms', 'MiB', 'GB/s', 'us/launch'))
for k, v in sorted(fam.items(), key=lambda kv: -kv[1][1]):
    if v[1] * 1e3 < 5: continue
    print('%-34s %7d %9.1f %9.0f %8s %8.1f' % (k, v[0], v[1]*1e3, v[2]/MiB, ('%.0f' % (v[2]/v[1]/1e9)) if v[2] else '-', v[1]/v[0]*1e6))
print('weight-reading kernels: %.3f s for %.2f GiB = %.0f GB/s (%.0f%% of 273); other kernels %.3f s' % (tw, bw/2**30, bw/tw/1e9, 100*bw/tw/BUS, to))
ideal = bw / BUS
print('bytes at 273 GB/s: %.3f s; at 240 GB/s: %.3f s' % (ideal, bw/240e9))
S = span/1e9
print('share of span: weight reads at bus speed %.1f%%, weight-read shortfall %.1f%%, other kernels %.1f%%, idle %.1f%%  (kernel overlap %.1f%%)'
      % (100*ideal/S, 100*(tw-ideal)/S, 100*to/S, 100*(span-busy)/1e9/S, 100*(tw+to-busy/1e9)/S))
if unk: print('unmapped:', dict(unk.most_common(8)))
