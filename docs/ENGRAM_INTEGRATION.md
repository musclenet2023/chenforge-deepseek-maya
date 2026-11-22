# Engram Integration — how the side-pack works

The "Engram" of DeepSeek-V4.1-Flash-1017 is a pair of n-gram lookup tables
(`blk.1` / `blk.14`, [256, ~384 M rows] each) that the engine consults during decode.
This document records, with engine-source evidence, how the packer integrates them and
what the engine accepts.

## What the engine accepts

| Aspect | Fact | Evidence (engine source, read-only) |
|---|---|---|
| Table dtypes | **Q6_K / IQ4_NL / MXFP8 / F8_E4M3 only** — no Q8_0 | dtype whitelist `ds4_model.cu:3163-3196`; decode kernels `ds4_kernels.cu:2172` (fmt 1=Q6_K, 2=IQ4_NL, 0=FP8) and `:2719` |
| IQ4_NL row layout | 32 values / 18 bytes: F16 scale + 16 nibble bytes; **no row padding** (144 B/row at head_dim 256) | `ds4_q6k.cuh:29-38` (matches ggml upstream `dequantize_row_iq4_nl`); row addressing `ds4_model.cu:6942` (`r.n = hd/32*18`) |
| Row divisibility | `eg_hd % 32 == 0` required (`ds4_model.cu:3189`) | 256 → 8 blocks, exact |
| RAM mode cache | `DS4_ENGRAM_CACHE_MB` (`ds4_model.cu:3138`) | row cache above NVMe mmap |
| DSE1 header | `<4s6i>` = magic, vocab, layers, ngram, heads, **head_dim**, pad | writer `pack_gguf.py` `write_engram()`; reader `ds4_model.cu:421-446` |
| Projection tensors | `engram.wkv` (Q8_0 [6144,25600]), `engram.q_weight`/`engram.k_weight` (BF16 [5120,4]); engine folds q/k itself | renames at `pack_gguf.py:129-134`; engine `ds4_model.cu:4567-4572` |

## Tensor-name and key aliases (packer)

| Source GGUF name | Engine/pack name |
|---|---|
| `blk.N.engram_embd.weight` | `layers.N.engram.embed.weight` |
| `engram_wkv.weight` | `engram.wkv.weight` |
| `engram_q.weight` / `engram_k.weight` | `engram.q_weight` / `engram.k_weight` |
| `per_layer_token_embd.weight` (single big tensor) | split per layer via `split_ple()`, needs `engram.num_embeddings` |
| `engram_layer_ids` / `engram_n_heads` / `engram_pad_token_id` | `engram.layers` / `engram.head_count` / `engram.pad_token_id` |
| `engram_key_length` or `engram.key_length` | `engram.head_dim` *(ChenForge alias fix)* |

## The Q8_0 problem and the IQ4_NL answer

Community IQ3_XXS quants carry the tables in-shard as **Q8_0**, which the engine cannot
read. Two validated resolutions:

1. **Use a side IQ4_NL set (recommended, zero conversion).** For DeepSeek-V4.1-Flash-1017
   a ready IQ4_NL Engram file exists (110.595 GB, SHA256
   `3dac4c54…9fb4f3`); sampling 64 rows showed Q8_0-dequant vs IQ4_NL-decode at
   **SNR 22.4 dB** — same f32 source table, two quantizations. The packer takes tables
   from `--engram <side>` and drops the shards' Q8_0 copies.
2. **Requantize Q8_0→IQ4_NL** (fallback when no side set exists). Algorithm validated on
   samples (SNR 17.3 dB with a simplified scale search; ggml's `ggml_quantize_chunk`
   would do at least as well). Streaming design: row-independent, ~1 GB RAM peak,
   ~1–3 h for 110 GB. **Never re-label Q8_0 bytes as IQ4_NL** — layouts differ
   (34 B/block vs 18 B/block) and the result decodes to garbage (demonstrated).

## What is *not* integrated

- A native Q8_0 Engram kernel for the engine (~30–60 lines: whitelist + fmt=3 decode
  branch). Drafted as an upstream suggestion; not built, not applied.
- MXFP8/FP8 side sets have not been exercised by this toolkit (engine supports them;
  untested paths are labeled untested in `KNOWN_LIMITATIONS.md`).
