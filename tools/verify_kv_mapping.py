"""Re-check the engine's KV-source invariant (ds4_model.cu) over every layer of a packed
ds4.txt.  Invariant: every layer with ratio > 0 must be preceded by a KV source of the same
ratio.

Usage:  python verify_kv_mapping.py <path-to-pack-dir-or-ds4.txt>
        (defaults to ./pack/ds4.txt next to this script's caller)
"""
import os
import sys


def read_geometry(path):
    geo = {}
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) >= 2:
            geo[parts[0]] = ",".join(parts[1:])
    return geo


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else os.path.join("pack", "ds4.txt")
    path = arg if arg.endswith("ds4.txt") else os.path.join(arg, "ds4.txt")
    geo = read_geometry(path)

    ratios = [int(x) for x in geo["compress_ratios"].split(",")]
    kv_src = [int(x) for x in geo["kv_source_layer_ids"].split(",")] if geo.get("kv_source_layer_ids") else []
    ix_src = [int(x) for x in geo["index_source_layer_ids"].split(",")] if geo.get("index_source_layer_ids") else []
    nl = int(geo["num_hidden_layers"])
    eg = [int(x) for x in geo["engram_layer_ids"].split(",")]
    cand = int(geo.get("candidate_source_layer_id", -1))

    print(f"pack: {path}")
    print(f"num_hidden_layers={nl} ratios({len(ratios)})={ratios}")
    print(f"kv_source={kv_src}  index_source={ix_src}  engram={eg}  candidate={cand}")
    assert len(ratios) >= nl, "ratios < layers"

    errors = []
    last_src_ratio = None
    for il in range(nl + 2):  # +2: simulate the mtp layer (ratio 0)
        r = ratios[il] if il < nl else 0
        is_src = il in kv_src
        if is_src:
            last_src_ratio = r
            print(f"  layer {il:2d}: KV source, ratio {r}")
        if r > 0:
            ok = last_src_ratio == r
            print(f"  layer {il:2d}: ratio {r} -> {'OK' if ok else 'FAIL'} (last source ratio={last_src_ratio})")
            if not ok:
                errors.append((il, r, last_src_ratio))

    for s in kv_src:
        if ratios[s] == 0:
            errors.append((s, "kv_source with ratio 0", ratios[s]))
    if errors:
        print("\nFAIL:", errors)
        raise SystemExit(1)
    print("\nALL LAYERS PASS: every ratio>0 layer has a KV source of the same ratio before it")


if __name__ == "__main__":
    main()
