# ChenForge v0.2.0 — Release Notes (draft)

**Focus: from two fixed quants to a reusable compatibility method.**

## Added

- **docs/COMPATIBILITY_METHOD.md** — the five principles distilled from the IQ2/IQ3 work
  (geometry lives in the pack, verify before run, derive from the checkpoint's own
  tensors, fail loudly, zero-copy packing) plus an Engram decision table and a
  step-by-step checklist for new quants.
- **docs/IQ2_COMPATIBILITY_JOURNEY.md / docs/IQ3_COMPATIBILITY_JOURNEY.md** — full
  engineering narratives with error text, root causes (engine file:line), before/after
  diffs, verification results, and explicit attribution: the pack-only method, zero-copy
  packing and the Engram side-pack contract were established on **IQ2_XXS**; the four
  metadata fallbacks (KV-source, candidate-geometry, key aliases, side-Engram with
  in-shard Q8_0) were new work on **IQ3_XXS**.
- **docs/ENGRAM_INTEGRATION.md** — the Engram contract: supported table dtypes
  (Q6_K/IQ4_NL/MXFP8/FP8 — never Q8_0), the 144 B/row IQ4_NL layout, all tensor/key
  aliases, and the two validated ways to resolve a Q8_0 Engram.
- **docs/KNOWN_LIMITATIONS.md** — every unverified item, stated plainly: 192K/256K
  never run, benchmark depth (single prompt × 3), requantization designed but not run,
  checkpoint-specific derivations.
- **docs/UPSTREAM_PR_DRAFTS.md** — overlap analysis against Niko1221/Strata main and a
  draft PR for the one fix genuinely missing upstream (tensor-derived geometry fallback).
- README: engineering-evolution table, compatibility matrix, common-errors → fixes map
  (English and 繁體中文).
- GitHub topics: deepseek, gguf, llama-cpp, quantization, local-llm, moe, compatibility,
  windows, cuda, huggingface.

## Changed

- Nothing in the fix set itself — `tools/pack_gguf_patched.py` and the patch document
  are unchanged from v0.1.0. No tag or release asset of v0.1.0 was modified.

## Verification status (unchanged from v0.1.0, restated)

- Verified: IQ2 and IQ3 packs produce `ALL LAYERS PASS`; both engines load and serve at
  96K ctx on 2× RTX 4090 + 128 GB RAM (IQ3: cold 15.8 s / warm 18.7–19.1 tok/s).
- Documented as design-only: Q8_0→IQ4_NL requantization; upstream engine defensive patch.

## Upstream

One PR candidate identified against Niko1221/Strata (tensor-derived geometry fallback);
aliases and side-Engram handling already exist upstream — no duplicate submissions.
Draft and diff in docs/UPSTREAM_PR_DRAFTS.md; **not submitted**.
