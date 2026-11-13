#!/usr/bin/env python3
"""verify.py <variant-dir>: every tensor outside the selected set must be byte-identical to stock."""
import glob, sys, hashlib, numpy as np
from gguf import GGUFReader
def load(d):
    m = {}
    for f in sorted(glob.glob(d + '/*.gguf')):
        for t in GGUFReader(f).tensors: m[t.name] = t
    return m
a = load('models/UD-Q4_K_XL'); b = load(sys.argv[1])
assert a.keys() == b.keys(), (len(a), len(b))
same = diff = 0; changed = {}
for n, t in a.items():
    u = b[n]
    if t.tensor_type == u.tensor_type:
        if np.array_equal(np.asarray(t.data).view(np.uint8).ravel(), np.asarray(u.data).view(np.uint8).ravel()): same += 1
        else: diff += 1; print('DIFFERENT BYTES, same type:', n)
    else:
        changed[(t.tensor_type.name, u.tensor_type.name)] = changed.get((t.tensor_type.name, u.tensor_type.name), 0) + 1
print('identical', same, 'same-type-but-different', diff, 'retyped', changed)
