"""Compose an Unsloth "mix" source tree locally.

The fork's release tags (e.g. b10715-mix-86bd2d3) are manifests, not source trees: the
prebuilt is assembled by CI as upstream tag + the PR commits pinned in
scripts/unsloth/pr-set.json, merged in order, with add/add conflicts resolved by their
additive_merge.py. This reproduces that, so a one-line patch can be built on exactly the
tree the prebuilt came from.

usage: compose-mix.py <upstream-tag> <pr-set.json> <additive_merge.py>   (run inside a full
       clone of ggml-org/llama.cpp; leaves branch "mix" checked out)
"""
import json, re, subprocess, sys
tag, prset = sys.argv[1], json.load(open(sys.argv[2]))["prs"]
def sh(*a, check=True):
    r = subprocess.run(a, capture_output=True, text=True); 
    if check and r.returncode: print("!!", " ".join(a), "\n", r.stdout[-800:], r.stderr[-800:]); sys.exit(1)
    return r
sh("git", "checkout", "-q", "-B", "mix", tag)
for url in prset:
    m = re.match(r"https://github.com/([^/]+/[^/]+)/pull/(\d+)/commits/([0-9a-f]+)", url)
    repo, pr, sha = m.group(1), m.group(2), m.group(3)
    sh("git", "fetch", "-q", f"https://github.com/{repo}", sha)
    r = sh("git", "merge", "--no-edit", "-q", sha, check=False)
    if r.returncode:
        a = subprocess.run(["python3", sys.argv[3], "--repo", "."], capture_output=True, text=True)
        if a.returncode: print(f"!! unresolved conflict merging {repo}#{pr} {sha[:8]}\n", a.stdout[-600:], r.stdout[-400:]); sys.exit(1)
        sh("git", "add", "-A"); sh("git", "-c", "core.editor=true", "commit", "-q", "--no-edit")
        print(f"merged {repo}#{pr} {sha[:8]} (additive conflict resolved)")
    else:
        print(f"merged {repo}#{pr} {sha[:8]}")
print("MIX COMPOSED", sh("git", "rev-parse", "--short", "HEAD").stdout.strip())
