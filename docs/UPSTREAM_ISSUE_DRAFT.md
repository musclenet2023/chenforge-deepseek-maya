# GitHub Issue — Niko1221/Strata

**Submitted as Strata issue #6640 (2026-11-23):**
https://github.com/Niko1221/Strata/issues/6640

**Title:** Tensor-derived geometry fallback for DeepSeek V4.1 Flash GGUF compatibility

---

### Summary

Community GGUF quants of DeepSeek-V4.1-Flash-1017 can omit sparse-attention metadata keys
(KV/index source layers, candidate geometry) from the GGUF header. The resulting pack then
has empty KV/index source lists in `ds4.txt`, and the engine exits at startup with:

```
maya-ds4: layer 2: compress ratio 2 without a KV source of that ratio before it
```

and — after fixing that by hand — crashes at the first forward pass with `0xC0000094`
(integer divide by zero) because `candidate_block_size` defaults to 0.

`apply_source_config()` / `--config-json` already covers this when the user has the HF
`config.json` at hand. We propose adding a **tensor-derived fallback** for when no config
is supplied: the missing layers are architecture, not quantization, and the shards prove it.

### IQ2 vs IQ3 quants — why one works and the other doesn't

| | IQ2_XXS (ISTA-DASLab GSQ-RCO, 6 shards) | IQ3_XXS (prometheusAIR, 10 shards) |
|---|---|---|
| `attention.kv_source_layers` / `attention.indexer.source_layers` | present in GGUF | **missing** |
| candidate keys (`candidate_source_layer`, `candidate_top_k_blocks`, `candidate_block_size`) | present in GGUF | **missing** |
| `engram.key_length` / `engram.pad_id` spellings | — | present (already aliased upstream, #3955 — not raising this again) |
| Engram tables | side IQ4_NL file | in-shard Q8_0 (engine has no Q8_0 Engram kernel; worked around packer-side) |

### Evidence (measured on the prometheusAIR IQ3_XXS set, 10 shards, 417 GB)

1. **Startup error** (above) — engine source: every `ratio > 0` layer must be preceded by a
   KV source of the same ratio (`ds4_model.cu:2671`, engine 1.1.30).
2. **First-forward crash**: WER capture, faulting offset `0x94778`; capstone disassembly
   shows `idiv ecx` with `ecx = g.cand_blk`, matching the ceil-division
   `(n[t] + g.cand_blk - 1) / g.cand_blk` (`ds4_model.cu:12508`; `cand_blk` defaults to 0 at
   `:502-503`). Confidence ~95% (no PDB symbols loadable; offsets and source align).
3. **The KV/index values are derivable from the shards themselves**: compressor tensors
   (`blk.N.attn_compressor_kv/gate/norm`) sit exactly on layers 2, 8, 14, 20; indexer tensors
   (`blk.N.indexer.*`) on 2, 8, 14, 20, 24, 28, 32, 36. Derived independently from the IQ3
   shards alone, they match the IQ2 quant's metadata for the same checkpoint exactly —
   the two quants differ only in metadata completeness.

Other quants of this checkpoint that omit these key groups are *likely* affected the same
way, but we have only measured the IQ3_XXS set above and claim nothing beyond it.

### Minimal reproduction (no weights shared)

- Measured: the prometheusAIR IQ3_XXS 10-shard set, packed with the stock packer, engine
  started → the startup error above; after hand-patching `kv_source_layer_ids`, exit
  `0xC0000094` at first forward.
- Quants other than IQ3_XXS may reproduce the same failure if they omit these keys, but we
  have not verified them. No weights need to be shared to follow along.

### Verified fix (packer-side, engine untouched)

In `geometry()` of `tools/dsv41/pack_gguf.py`, after the existing KV block:

```python
# community quants leave the sparse-attention geometry out of the metadata; the layers
# are architecture, not quantization: derive them from the shards' own tensors.
# NOTE: distinguish "key absent" from "key present but zero/empty" — a legitimately
# empty list or a 0 value must not be silently overwritten by the fallback.
KV_MISSING = "kv_source_layer_ids" not in out
IX_MISSING = "index_source_layer_ids" not in out
if KV_MISSING:
    out["kv_source_layer_ids"] = sorted({t[0].split(".")[1] for t in tensors
                                         if ".attn_compressor_" in t[0]}, key=int)
if IX_MISSING:
    out["index_source_layer_ids"] = sorted({t[0].split(".")[1] for t in tensors
                                            if ".indexer." in t[0]}, key=int)
```

**Candidate geometry is the exception**: the KV/index lists are provably derivable (tensors
exist on exactly those layers), but we found no tensor-derived rule for the `candidate_*`
values. We did **not** hardcode them. In our run they were filled from a cross-checked
sibling quant of the same checkpoint — validated as a sibling by confirming the base
checkpoint and architecture actually match (identical `hidden_size`, `num_hidden_layers`,
`head_dim`, MoE configuration and `compress_ratios` layout, not just a similar model name):
IQ2_XXS of DeepSeek-V4.1-Flash-1017 gives candidate source 20 / topk 2048 / block 8.

The proposed patch fills candidate values only when a validated sibling or explicit
`--config-json` supplies them, treats key-absent, present-but-zero, and legitimate values
as distinct states, and otherwise exits with a loud error listing exactly which keys could
not be resolved. `--config-json` stays authoritative whenever given.

Result: pack diff is exactly 5 lines of `ds4.txt`; an all-layer re-implementation of the
KV invariant reports ALL LAYERS PASS; the engine then loads and serves at 96K context on
2x RTX 4090 + 128 GB RAM (cold 15.8 s / TTFT 3.9 s, warm 18.7-19.1 tok/s). All measured on
the IQ3_XXS set above.

### Questions for the maintainers

1. Would you accept a tensor-derived KV/index geometry fallback (plus the cross-checked
   candidate path described above) as a patch to `tools/dsv41/pack_gguf.py`, in addition to
   the existing `--config-json` path?
2. If yes — where should it target, given the DeepSeek engine was removed from
   builds/packaging on 2026-11-11 ("DeepSeek out (7/7)")? Is the packer still maintained
   in-tree, or has this work moved to a different project/fork?
3. Separately: rather than masking the bad value, would you accept an engine-side change
   that **validates candidate geometry explicitly and errors out with a clear message**
   when `candidate_block_size` is zero or the candidate source layer is unset — so that
   packer/config mistakes surface as a readable error instead of a `0xC0000094` crash?
   Happy to open a dedicated issue if that's still in scope.

### Links

- Full fix set, patches, verification tools and journey docs:
  https://github.com/musclenet2023/chenforge-deepseek-maya (v0.2.0)
- Data mirror (matrix, ds4.txt examples, verification log, benchmark):
  https://huggingface.co/datasets/musclenet2023/chenforge-deepseek-v4-flash-compatibility
- Upstream overlap analysis + diff draft:
  https://github.com/musclenet2023/chenforge-deepseek-maya/blob/main/docs/UPSTREAM_PR_DRAFTS.md

- Upstream: issue #6640 filed at Niko1221/Strata asking whether the tensor-derived
  geometry fallback is wanted and where it should target, given the DeepSeek engine was
  removed from upstream builds on 2026-11-11:
  https://github.com/Niko1221/Strata/issues/6640

*No model weights are distributed by that repository; all benchmarks above were run on
local hardware and are documented with methodology in the repo.*
