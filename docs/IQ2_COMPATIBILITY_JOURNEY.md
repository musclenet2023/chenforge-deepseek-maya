# IQ2_XXS Compatibility Journey — the first quant

This is the chronological record of making the **first** community DeepSeek-V4.1-Flash-1017
quant (the GSQ-RCO **IQ2_XXS** 6-shard set + IQ4_NL Engram) run on the Maya DS4 engine.
All fixes below were **first developed here**; the IQ3_XXS work
(`IQ3_COMPATIBILITY_JOURNEY.md`) reused this method and only added what IQ3's different
metadata layout required. Credit follows the work, not the later quant.

Evidence sources: local pack logs, `ds4.txt`/`tensors.txt` packer products, engine source
`src/ds4/ds4_model.cu` (read-only file:line references), and a measured benchmark
(`docs/BENCHMARK.md`). No claim below is extrapolated without saying so.

## 1. Baseline: what IQ2_XXS is

| Item | Value (measured from GGUF headers + pack products) |
|---|---|
| Quant | Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF (6 shards) |
| Architecture | `deepseek41`, 40 layers, 384 experts, top-6, ctx 1,048,576 |
| Engram | side file `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`, 110.595 GB, IQ4_NL, 144 B/row |
| Pack output | 48,531 tensors / 12 files; weights never copied (packer writes metadata only) |

IQ2_XXS was the *easier* case metadata-wise: its GGUF header natively carries
`deepseek41.attention.kv_source_layers`, `attention.indexer.source_layers` and the three
candidate keys. The hard part was **proving the pipeline end to end** — download
integrity, shard-6 hardlinking, pack, geometry verification, and a real benchmark.

## 2. What the IQ2 work established (reused unchanged by IQ3)

1. **The pack-only method.** The engine reads geometry from the pack product `ds4.txt`
   (`ds4_model.cu:354`), not from GGUF metadata. So every metadata problem is a *packer*
   problem — no engine change and no 400 GB rewrite is ever needed. This single insight
   is the core reusable method of the whole toolkit.
2. **Zero-copy packing.** `pack_gguf.py` opens every source GGUF `'rb'`-only (verified:
   6 `.open()` call sites) and the output `tensors.txt` references shards by relative path.
   A second quant can share hardlinked files (NTFS `fsutil hardlink`) with zero data copy —
   the IQ3 pack reused the 110 GB IQ4_NL Engram file this way (`st_nlink = 3`).
3. **Engram side-pack contract.** `engram.bin` = DSE1 header (`<4s6i>`) + token map +
   hash parameters; the engine reader is `ds4_model.cu:421-446`. IQ2's `ple.*` metadata
   spelling fills `engram.head_dim` from `embedding_length_per_layer_input` via
   `ple_to_engram()`.
4. **Verification tool.** `tools/verify_kv_mapping.py` re-implements the engine's
   last-source invariant (`ds4_model.cu:2671`) over all 40 layers + MTP and reports
   `ALL LAYERS PASS` — a preflight that does not need the engine.
5. **Measured benchmark** (2× RTX 4090 48 GB, 128 GB RAM, 96K ctx): cold 15.8 s wall /
   3.9 s TTFT / 7.9 tok/s; warm ~5.8–6.0 s / 0.48–0.77 s TTFT / 18.7–19.1 tok/s;
   expert cache hit 59% → 79% across three runs. Full numbers in `docs/BENCHMARK.md`.

## 3. What IQ2 did *not* need

Because the IQ2 GGUF carries the KV/index/candidate keys natively, none of the metadata
fallbacks in `patches/pack_gguf_ds41-compat.patch.md` were exercised by IQ2. Those four
fixes exist because **IQ3 lacked the keys** — see the IQ3 journey. This distinction is
the reason the patch file says "tested against IQ2 and IQ3" but the *fallback* code
paths were proven on IQ3.

## 4. Unverified / open items (stated, not hidden)

- IQ2's `download_hf_model.py` path (download → SHA256 → shard-6 hardlink) is validated
  as a workflow but was not re-run against a fresh download during the v0.1.0 work.
- 192K/256K contexts were never run on this machine (KV ~8 GB at 192K fits, but
  RAM headroom at 128 GB makes 96K the tested maximum).
- The IQ2 pack benchmark is a single-prompt × 3-run sample; it is not a suite.
