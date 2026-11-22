Minimal patch: pack_gguf.py — Maya DS4 compatibility fixes for third-party DeepSeek V4.1 Flash quants
=====================================================================================================

Applies to: Maya / Strata `tools/ds4/pack_gguf.py` (MIT, Maya_DS4_Test)
Tested against: packer output 48,531 tensors / 12 files, IQ2_XXS and IQ3_XXS quants of
DeepSeek-V4.1-Flash-1017 (deepseek41 arch, 40 layers, 384 experts, top-6).

Problem
-------
Community quantizations (e.g. the prometheusAIR IQ3_XXS set) omit several metadata keys the
engine's geometry loader requires:

1. `deepseek41.attention.kv_source_layers` and
   `deepseek41.attention.indexer.source_layers` — the engine requires every layer with
   `compress_ratios > 0` to be preceded by a KV source of the *same* ratio
   (ds4_model.cu:2671). Missing keys produce:

       maya-ds4: layer 2: compress ratio 2 without a KV source of that ratio before it

2. `deepseek41.attention.indexer.candidate_source_layer / candidate_top_k_blocks /
   candidate_block_size` — the engine defaults to `cand_blk = 0` (ds4_model.cu:502-503) and
   the first forward crashes with an integer divide-by-zero (0xC0000094).

3. `engram.key_length` (the prometheus quant spells the Engram head dim `key_length`, not
   `head_dim`) and `engram.pad_id` — `write_engram()` then packs None and crashes.

4. Prometheus carries each layer's Engram table as its own `blk.N.engram_embd.weight`
   (Q8_0) and lacks `engram.num_embeddings`, so `split_ple()` cannot split a side set's
   `per_layer_token_embd`. The engine has **no Q8_0 Engram kernel** (only Q6_K / IQ4_NL /
   MXFP8 / FP8, ds4_model.cu:3163-3196), so the shards' own Q8_0 tables must be dropped
   in favour of a side IQ4_NL set passed with `--engram`.

Fix
---
All fixes are packer-side; the engine is not modified.

```diff
--- a/tools/ds4/pack_gguf.py
+++ b/tools/ds4/pack_gguf.py
@@ -193,6 +193,26 @@
     if clamp and any(abs(c - clamp[0]) > 1e-6 for c in clamp):
         raise SystemExit("swiglu_clamp_exp differs between layers %s: the engine takes one swiglu_limit" % clamp[:8])
+    # the quant left the KV source keys out of the metadata: the layers with a compressor read
+    # the KV a ratio-2 source holds (and the indexer's the ratio-1 one), so take them from the
+    # shards' own tensors.  The engine requires every layer of ratio > 0 to have a source of the
+    # same ratio before it (ds4_model.cu).
+    if not out["kv_source_layer_ids"]:
+        out["kv_source_layer_ids"] = kv_layers(kv)  # the compressor layers: ratio 2
+    if not out["index_source_layer_ids"]:
+        out["index_source_layer_ids"] = ix_layers(kv)   # the indexer's: any of the 1s
+    # the quant also left the indexer's candidate keys out of the metadata - the engine defaults
+    # to cand_src=-1 / cand_topk=0 / cand_blk=0 (ds4_model.cu:502-503) and the first forward then
+    # divides by cand_blk = 0 (0xC0000094).  The values are architecture, not quantization:
+    # take them from the IQ2_XXS pack of the same model, whose GGUF carries
+    # deepseek41.attention.indexer.candidate_source_layer=20, candidate_top_k_blocks=2048,
+    # candidate_block_size=8.  With cand_src=20 one of the kv_source layers (ratio 2) feeds the
+    # candidate path.
+    if not out["candidate_source_layer_id"]:
+        out["candidate_source_layer_id"] = 20
+        out["candidate_topk_blocks"] = out["candidate_topk_blocks"] or 2048
+        out["candidate_block_size"] = out["candidate_block_size"] or 8
+        print("candidate geometry not in the GGUF metadata, taken from the IQ2_XXS pack of the same model: "
+              "candidate_source_layer_id 20, candidate_topk_blocks 2048, candidate_block_size 8")
     return out
+
+
+def kv_layers(kv: dict):
+    """The layers holding a compressor KV projection (blk.N.attn_compressor_kv / _gate / _norm):
+    the ratio-2 sources.  From the shards' tensor lists - the quant's GGUF gives no kv_source
+    metadata."""
+    a = kv["general.architecture"]
+    src = {int(m.group(1)) for g in (GS or []) for (nm, *_r) in g.tensors
+           if (m := re.match(r"blk\.(\d+)\.attn_compressor_(?:kv|gate|norm)\.weight$", nm))}
+    if not src:
+        raise SystemExit("%s.attention.kv_source_layers is not in the metadata and no compressor "
+                         "tensors were found - cannot derive the KV sources" % a)
+    return sorted(src)
+
+
+def ix_layers(kv: dict):
+    """The layers holding an indexer (blk.N.indexer.*): the sources of the ratio-1 (uncompressed)
+    KV."""
+    src = {int(m.group(1)) for g in (GS or []) for (nm, *_r) in g.tensors
+           if (m := re.match(r"blk\.(\d+)\.indexer\.(?:attn_q_b|attn_k|k_norm|proj)\.weight$", nm))}
+    if not src:
+        raise SystemExit("no indexer tensors were found - cannot derive the index sources")
+    return sorted(src)
@@ NEWER_KEYS (alias table)
     ("engram_layer_ids", "engram.layers"), ("engram_n_heads", "engram.head_count"),
+    ("engram_key_length", "engram.head_dim"), ("engram.key_length", "engram.head_dim"),
+    ("engram.pad_id", "engram.pad_token_id"),
@@ module level
+GS = None   # the shards' tensor lists (set in main): kv_layers / ix_layers read them
@@ main()
+    global GS
+    GS = gs
@@ split_ple() — fill num_embeddings from the shards' own per-layer tables
+    if not kv.get("%s.engram.num_embeddings" % kv["general.architecture"]):
+        rows = {t[0]: t[2][1] for g in gs for t in g.tensors if t[0].endswith(".engram_embd.weight")}
+        fill = [rows.get("blk.%d.engram_embd.weight" % il)
+                for il in kv.get("%s.engram.layers" % kv["general.architecture"], [])]
+        if all(fill):
+            kv["%s.engram.num_embeddings" % kv["general.architecture"]] = fill
@@ main(), after --engram is parsed — drop the shards' own Q8_0 Engram tables
+    # with --engram, drop the shards' own Engram tables so the side set's IQ4_NL ones are picked
+    # instead - the engine has no Q8_0 Engram kernel; the q/k/gate projections stay the quant's
+    if a.engram:
+        for g in gs:
+            g.tensors = [t for t in g.tensors if ".engram_embd." not in t[0]]
```

`tools/pack_gguf_patched.py` in this repository is the full patched copy; the diff above is
the minimal change set against stock Maya/Strata `tools/ds4/pack_gguf.py`.

Candidate values: if you cannot obtain the IQ2_XXS pack's ds4.txt, the values
(20 / 2048 / 8) are the DeepSeek-V4.1-Flash architecture's own indexer geometry —
verifiable from any quant of the same checkpoint that carries the candidate keys, or from
the compressor/indexer tensor layout (the compressor sits on layers 2/8/14/20, so the
candidate source is 20).

Verification
------------
- `tools/verify_kv_mapping.py` re-checks the engine's invariant over all layers from the
  produced `ds4.txt`: ALL LAYERS PASS (kv_source=[2,8,14,20], index_source=[2,8,14,20,24,28,32,36],
  engram=[1,14], candidate=20 — identical to the IQ2_XXS pack of the same checkpoint).
- Packer re-run output: 48,531 tensors / 12 files, same capacity distribution.
- Server startup and first forward with the fixed pack: see docs/TRoubleshooting.md.
