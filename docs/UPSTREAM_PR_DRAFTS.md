# Upstream PR drafts — for review, NOT yet submitted

Target repository: **Niko1221/Strata** (`tools/dsv41/pack_gguf.py`; MIT).
Upstream state checked 2026-11-23 at commit `86c94204a1` (2026-10-31); note the
2026-11-11 commit "DeepSeek out (7/7)" indicates the DeepSeek *engine* was removed from
upstream builds/packaging — PRs should target the packer tool, and we should ask the
maintainer where DS4 packer fixes are still welcome.

## Overlap check (what upstream already has)

| Our fix | Upstream status |
|---|---|
| Fix 3 (`engram.key_length` / `engram.pad_id` aliases) | **Already upstream** (`engram.head_dim ↔ engram.key_length` alias lines, `fb966cff37` "accept older DeepSeek GGUF metadata names" #3955). Do **not** PR. |
| Fix 4a (`num_embeddings` missing) | Upstream `split_ple()` handles layered `blk.N.engram_embd` reads and fails loudly; partially covered. Re-check before PR. |
| Fix 1/2 (KV-source + candidate fallbacks) | Upstream has `apply_source_config()` but it requires the user to pass a HF `config.json` (`--config-json`); **no tensor-derived fallback** and no candidate default when no config is given. Our fix is additive and independent. |
| Fix 4 (drop in-shard Q8_0 Engram when `--engram` given) | Upstream picks/drops `engram_embd` parts when a side set is given (lines ~557–587); appears covered. Verify against current main before PR. |

## PR 1 — tensor-derived KV/index/candidate geometry fallback

One PR, ~25 lines + tests, independent of the config-json path.

```diff
--- a/tools/dsv41/pack_gguf.py
+++ b/tools/dsv41/pack_gguf.py
@@ geometry(), after the KV block:
+    # community quants (e.g. prometheusAIR IQ3_XXS) leave the sparse-attention geometry
+    # out of the GGUF metadata entirely.  the layers are architecture, not quantization:
+    # derive them from the shards' own tensors when no config.json is supplied.
+    if not out["kv_source_layer_ids"]:
+        out["kv_source_layer_ids"] = sorted({t[0].split(".")[1] for t in tensors
+                                             if ".attn_compressor_" in t[0]}, key=int)
+    if not out["index_source_layer_ids"]:
+        out["index_source_layer_ids"] = sorted({t[0].split(".")[1] for t in tensors
+                                               if ".indexer." in t[0]}, key=int)
+    if not out["candidate_source_layer_id"]:
+        # defaults cross-checked against a quant of the same checkpoint that carries
+        # them (IQ2_XXS of DeepSeek-V4.1-Flash-1017: source 20 / topk 2048 / block 8)
+        out["candidate_source_layer_id"] = 20
+        out["candidate_topk_blocks"] = out["candidate_topk_blocks"] or 2048
+        out["candidate_block_size"] = out["candidate_block_size"] or 8
+    if not out["kv_source_layer_ids"]:
+        raise SystemExit("no KV source layers in the metadata or the tensors")
```

**PR body (draft):**

> Title: pack_gguf: derive sparse-attention geometry from the shards when the quant omits it
>
> Community IQ3_XXS quants of DeepSeek-V4.1-Flash-1017 (prometheusAIR) omit
> `attention.kv_source_layers`, `attention.indexer.source_layers` and the three candidate
> keys from the GGUF header. With the current packer the produced `ds4.txt` has empty
> source lists, the engine exits with `layer 2: compress ratio 2 without a KV source of
> that ratio before it`, and — after patching that by hand — the first forward crashes
> with 0xC0000094 because `candidate_block_size` defaults to 0.
>
> `--config-json` already covers this when the user has the HF config handy, but the
> values are derivable from the shards themselves: compressor tensors sit exactly on the
> ratio-2 source layers, indexer tensors on the ratio-1 ones (verified: derived
> [2,8,14,20] / [2,8,14,20,24,28,32,36] independently, matching the IQ2_XXS quant's
> metadata for the same checkpoint). This patch adds the tensor-derived fallback, keeps
> `--config-json` authoritative when given, and fails loudly when neither source can
> answer.
>
> Tested: prometheusAIR IQ3_XXS 10-shard set, pack 48,531 tensors / 12 files,
> all-layer KV invariant check passes, engine loads and serves at 96K ctx on 2×RTX 4090.

## PR 2 — (future, engine) defensive candidate divide

Not drafted as code here: `max(1, g.cand_blk)` at the three division sites
(engine `ds4_model.cu` `:12508/:13003/:6774`) or skipping the candidate path when
`cand_src == -1`. Worth an *issue* first — the engine was de-published upstream on
2026-11-11, so ask where engine fixes belong now.

## Process

Before any submission: re-clone upstream main, re-verify the overlap table above line by
line (the file is actively changing), then submit PR 1 only. No multi-PR split is needed
— only one of our four fixes is genuinely missing upstream.
