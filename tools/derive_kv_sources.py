"""Derive the KV-source / index-source layers of a DeepSeek V4.1 Flash GGUF set from its own
tensor list, without trusting the quant's metadata.

Cross-checks the derivation against the geometry the same checkpoint's IQ2_XXS quant carries
in its GGUF metadata (kv_source [2,8,14,20], index_source [2,8,14,20,24,28,32,36]).

Usage:  python derive_kv_sources.py <model-dir>
        where <model-dir> holds the GGUF shards; header-only scan (no weight data is read)
"""
import glob
import os
import re
import struct
import sys

# ggml type -> (name, values a block, bytes a block) — only what header parsing needs
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 30: ("BF16", 1, 2),
         10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
         13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66),
         17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 39: ("MXFP4", 32, 17),
         67: ("MXFP8", 32, 33), 43: ("MXFP8", 32, 33)}


def read_gguf_header(path):
    """Return (metadata dict, tensor-name list) from a GGUF file's header only."""
    f = open(path, "rb")
    magic = f.read(4)
    if magic != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    ver, n_kv, n_tensors = struct.unpack("<IqQ", f.read(20))
    str_, val = [], []

    def s():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8")

    kv = {}
    for _ in range(n_kv):
        k = s()
        t = struct.unpack("<I", f.read(4))[0]
        if t == 13:
            kv[k] = s()
        elif t in (2, 3, 4, 5, 6, 7, 8, 9, 10):
            kv[k] = struct.unpack("<q" if t == 7 else "<i", f.read(8 if t == 7 else 4))[0]
        elif t == 11:
            kv[k] = struct.unpack("<d", f.read(8))[0]
        elif t == 12:
            kv[k] = struct.unpack("<f", f.read(4))[0]
        elif t == 14:  # array: [elem-type, count, then elements]
            et, n = struct.unpack("<IQ", f.read(12))
            if et == 13:
                kv[k] = [s() for _i in range(n)]
            else:  # numeric array: skip by element size
                size = {0: 4, 1: 2, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 8, 8: 4, 9: 4, 10: 4,
                        11: 8, 12: 4}.get(et)
                if size is None:
                    raise SystemExit(f"unsupported array elem type {et}")
                f.seek(n * size, 1)
                kv[k] = f"<array:{et}x{n}>"
        else:
            raise SystemExit(f"unsupported kv value type {t}")

    names = []
    for _ in range(n_tensors):
        nm = s()
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        typ = struct.unpack("<I", f.read(4))[0]
        _off = struct.unpack("<Q", f.read(8))[0]
        if typ not in TYPES:
            raise SystemExit(f"{nm}: unknown type id {typ}")
        names.append((nm, typ, dims))
    f.close()
    return kv, names


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(d, "*.gguf")))
    if not files:
        raise SystemExit(f"no GGUF files under {d}")
    comp, ix, dims = set(), set(), {}
    for p in files:
        print(f"scanning {os.path.basename(p)} ...", flush=True)
        _kv, tensors = read_gguf_header(p)
        for nm, typ, dims_t in tensors:
            m = re.match(r"blk\.(\d+)\.attn_compressor_(?:kv|gate|norm)\.weight$", nm)
            if m:
                comp.add(int(m.group(1)))
                dims[nm] = dims_t
            m = re.match(r"blk\.(\d+)\.indexer\.(?:attn_q_b|attn_k|k_norm|proj)\.weight$", nm)
            if m:
                ix.add(int(m.group(1)))

    print("compressor layers :", sorted(comp))
    print("indexer layers    :", sorted(ix))
    print("IQ2 kv_source     : [2, 8, 14, 20]")
    print("IQ2 index_source  : [2, 8, 14, 20, 24, 28, 32, 36]")
    ok = sorted(comp) == [2, 8, 14, 20] and sorted(ix) == [2, 8, 14, 20, 24, 28, 32, 36]
    print("MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
