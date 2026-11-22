# The Compatibility Method — reusable across GGUF quants

This is the general procedure distilled from the IQ2_XXS and IQ3_XXS work. It applies to
any community GGUF quant of DeepSeek-V4.1-Flash-1017 (or another `deepseek41`-family
checkpoint) that the Maya DS4 engine refuses to load. Nothing here requires modifying the
engine or rewriting weights.

## Principle 1 — the engine reads geometry from the pack, not from the GGUF

`ds4_model.cu:354` (`std::ifstream f(d + "/ds4.txt")`) is the single entry point for
geometry: `compress_ratios`, `kv_source_layer_ids`, `index_source_layer_ids`, the three
candidate keys, `engram.*`. Consequences:

- Any metadata problem is a **packer** problem with a packer fix.
- Fixing it never touches the 200–400 GB shards.
- The failure mode is always readable in the produced `ds4.txt` — diff it.

## Principle 2 — verify before you run

`tools/verify_kv_mapping.py` re-implements the engine's invariant
(`ds4_model.cu:2671`: every `ratio > 0` layer must be preceded by a KV source of the
same ratio) over all layers + MTP, reading only `ds4.txt`. Run it after every pack.
`ALL LAYERS PASS` is the gate for even trying to start the engine.

**Preflight order** (recommended before spending any disk or hours):

1. `tools/derive_kv_sources.py` — header-only scan of the shards: derives KV/index
   sources from tensor placement, no download of full data needed beyond headers.
2. `tools/verify_kv_mapping.py <pack>` — after packing.
3. Only then start the engine.

## Principle 3 — derive missing values from the checkpoint's own tensors

When a quant's GGUF lacks a key, the value is usually **architecture, not
quantization**, and the tensors prove it:

- Compressor tensors (`blk.N.attn_compressor_*`) exist exactly on the ratio-2 source
  layers → `kv_source_layer_ids` = [2, 8, 14, 20].
- Indexer tensors (`blk.N.indexer.*`) → `index_source_layer_ids` =
  [2, 8, 14, 20, 24, 28, 32, 36].
- Candidate keys: cross-check against another quant of the *same checkpoint* that
  carries them (IQ2_XXS for DeepSeek-V4.1-Flash-1017), after confirming identical
  `compress_ratios`.

Never copy values between different checkpoints; always derive or cross-check.

## Principle 4 — fail loudly, never silently

Every fallback in `tools/pack_gguf_patched.py` ends in `SystemExit` when neither the
metadata nor the tensors can supply a value. A silent wrong guess in `ds4.txt` would
either crash the engine confusingly or — worse — run with wrong geometry.

## Principle 5 — zero-copy packing

- The packer opens all sources `'rb'`-only (6 `.open()` sites, all verified).
- Output references shards by relative path; a second quant can hardlink shared files
  (`fsutil hardlink create`) — e.g. the 110 GB IQ4_NL Engram file is shared between
  the IQ2 and IQ3 packs with `st_nlink = 3` and zero data copy.
- Disk cost of a second quant's pack ≈ tens of MB (metadata, tokenizer, index files).

## Decision table: Engram tables

| Situation | Action |
|---|---|
| Shards have no Engram tables; side IQ4_NL/Q6_K set exists | `--engram <side>` — engine-supported types directly |
| Shards carry Q8_0 Engram tables in-shard | packer drops them (`--engram` mode) and takes the side set |
| Only Q8_0 tables exist anywhere | design decision: requantize Q8_0→IQ4_NL (validated algorithm, streaming, SNR ≈ 22 dB vs Q8_0 reference) **or** add a Q8_0 Engram kernel to the engine (~30–60 lines, requires recompile) |
| `engram.head_dim` missing / misspelled | alias `engram.key_length` → `engram.head_dim` (and `engram.pad_id` → `engram.pad_token_id`) |

## Known engine-side gaps (documented, not patched by this toolkit)

- `cand_blk = 0` default with unguarded integer division (`ds4_model.cu:502-503`,
  `:12508`) — the packer fallback prevents it, but a defensive engine fix
  (`max(1, cand_blk)` or skip candidate path when `cand_src == -1`) would remove the
  whole failure class.
- No Q8_0 Engram kernel — see table above.

## Applying the method to a new quant (checklist)

1. Header-scan the shards (types, dims, which `deepseek41.*` keys exist).
2. Diff the key set against the reference quant of the same checkpoint.
3. Pack with `tools/pack_gguf_patched.py`; watch for fallback log lines.
4. `verify_kv_mapping.py` → ALL LAYERS PASS.
5. Start the engine; if it crashes, capture exit code + WER offset, map to source
   (the crash-to-source workflow is in `docs/TROUBLESHOOTING.md`).
6. Record what was new in a journey doc — attribution follows the quant that first
   needed each fix.
