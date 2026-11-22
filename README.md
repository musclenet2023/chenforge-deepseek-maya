# ChenForge — DeepSeek V4.1 Flash Compatibility Toolkit

Minimal, reproducible packer-side fixes that let the [Maya DS4 engine](https://github.com/Niko1221/Strata)
(MIT-licensed `deepseek41` architecture runtime) load community quantizations of
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) that
omit metadata the engine requires.

> **What this repository is:** four small fixes to `tools/ds4/pack_gguf.py` (the packer only —
> the engine is not modified), verification tools, and Windows example scripts.
> **What this repository is not:** it distributes no model weights, no GGUF files, and no
> engine binaries.

## The problem

Community GGUF quants of DeepSeek-V4.1-Flash-1017 (e.g. the prometheusAIR **IQ3_XXS** set)
leave several `deepseek41.*` metadata keys out of the GGUF header. The Maya DS4 engine's
geometry loader then fails at startup or at the first forward pass:

| Missing key(s) | Symptom | Root cause (engine source) |
|---|---|---|
| `attention.kv_source_layers`, `attention.indexer.source_layers` | `layer 2: compress ratio 2 without a KV source of that ratio before it` | every `ratio > 0` layer must be preceded by a KV source of the same ratio (`ds4_model.cu:2671`) |
| `attention.indexer.candidate_source_layer / candidate_top_k_blocks / candidate_block_size` | first-forward crash, exit `0xC0000094` (integer divide by zero) | engine defaults `cand_blk = 0` (`ds4_model.cu:502-503`) |
| `engram.head_dim` spelled `engram.key_length`, `engram.pad_id` | `write_engram()` packs `None`, packer crash | alias table lacked both spellings |
| `engram.num_embeddings` (per-layer `blk.N.engram_embd.weight` layout) | `split_ple()` cannot split a side Engram set | dims not derivable from metadata alone |

Additionally, the IQ3_XXS set carries its Engram tables as **Q8_0** — the engine has no Q8_0
Engram kernel (only Q6_K / IQ4_NL / MXFP8 / FP8, `ds4_model.cu:3163-3196`) — so the pack must
take the tables from a side **IQ4_NL** set via `--engram` and drop the shards' own Q8_0 tables.

## The fixes (packer-side only)

`patches/pack_gguf_ds41-compat.patch.md` documents the full minimal change set; `tools/pack_gguf_patched.py`
is the complete patched copy. In summary:

1. **KV-source fallback** — when the metadata lacks `kv_source_layer_ids` / `index_source_layer_ids`,
   derive them from the shards' own tensor list (compressor tensors ⇒ ratio-2 sources on layers
   2/8/14/20; indexer tensors ⇒ ratio-1 sources on layers 2/8/14/20/24/28/32/36). Fails loudly
   (`SystemExit`) when neither metadata nor tensors can supply the answer — never silently.
2. **Candidate-geometry fallback** — when the candidate keys are missing, take the
   architecture's own values (source layer 20, top-k blocks 2048, block size 8), cross-checked
   against the IQ2_XXS quant of the same checkpoint.
3. **Engram key aliases** — accept `engram.key_length` as `engram.head_dim` and
   `engram.pad_id` as `engram.pad_token_id`.
4. **Side-Engram support** — fill `engram.num_embeddings` from the shards' per-layer table dims,
   and with `--engram` drop the shards' own Q8_0 Engram tables so the engine reads the
   IQ4_NL side set instead.

## Repository layout

```
patches/    pack_gguf_ds41-compat.patch.md   the minimal diff, annotated (file:line evidence)
tools/      pack_gguf_patched.py             full patched packer copy
            verify_kv_mapping.py             re-checks the engine's KV invariant on a produced ds4.txt
            derive_kv_sources.py             header-only scan: derive KV/index sources from GGUF shards
            download_hf_model.py             download + SHA256-verify + shard-6 hardlink helper
examples/   example_pack_iq3.bat             Windows packing example (IQ3_XXS)
            example_pack_iq2.bat             Windows packing example (IQ2_XXS 6-shard)
docs/       INSTALL.md  BENCHMARK.md  TROUBLESHOOTING.md
LICENSE     MIT (this repository's fix set)
```

## Quick start (Windows)

```bat
git clone <this repo>
cd deepseek-v4-flash-maya-ds4-windows

:: 1. download the quant (example: the IQ2_XXS uncensored set) and verify SHA256
python tools\download_hf_model.py Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF D:\models\DS41-IQ2XXS

:: 2. pack for the Maya DS4 engine (weights are not copied; only metadata is written)
examples\example_pack_iq2.bat        :: edit MODEL_DIR first

:: 3. verify the geometry the engine will read
python tools\verify_kv_mapping.py D:\models\DS41-IQ2XXS\pack
```

See docs/INSTALL.md for prerequisites and disk requirements.

## Status

- IQ2_XXS (6-shard, Abiray uncensored GSQ-RCO): packed, geometry verified, server load tested.
- IQ3_XXS (10-shard, prometheusAIR): packed, geometry verified (ALL LAYERS PASS),
  server startup and 128-token generation tested at 96K context on 2× RTX 4090 + 128 GB RAM.
- Benchmark numbers: docs/BENCHMARK.md. Known failure modes and fixes: docs/TROUBLESHOOTING.md.

## Credits & licenses

- **DeepSeek AI** — DeepSeek-V4.1 architecture and base weights (MIT, `LICENSE.deepseek-v41` upstream).
- **ISTA-DASLab** — GSQ-RCO quantization pipeline and the shared `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` Engram backbone (the IQ2_XXS GGUF set is MIT-labeled).
- **Project Maya / Niko1221 (Strata)** — the DS4 engine and the original `tools/ds4/pack_gguf.py` (MIT).
- **prometheusAIR** — the IQ3_XXS quantization whose missing metadata motivated fixes 1-4.
- This repository's own fix set: MIT. Model weights stay under their upstream licenses and are
  **not** redistributed here.
