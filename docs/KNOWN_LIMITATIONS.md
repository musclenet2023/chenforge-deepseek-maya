# Known Limitations & Unverified Items

Everything below is explicitly **not verified** or is a known gap. Nothing on this page
should be quoted as a working feature.

## Unverified items

| Item | Status |
|---|---|
| `download_hf_model.py` fresh-download path | Workflow validated once; not re-run end-to-end on a clean machine during v0.1.0/v0.2.0 work |
| 192K / 256K contexts | Never run on the test machine (128 GB RAM); 96K is the tested maximum. KV ~8 GB at 192K fits arithmetically but was not exercised |
| IQ2_XXS benchmark depth | Single prompt × 3 runs (one cold, two warm). Not a benchmark suite; treat as smoke-test-grade |
| Q8_0→IQ4_NL full requantization | Algorithm validated on 64-row samples (SNR 17.3–22.4 dB); the full 110 GB conversion was never run (unnecessary — a ready IQ4_NL side file exists) |
| MXFP8 / F8_E4M3 Engram side sets | Engine-supported per source, never exercised by this toolkit |
| `derive_kv_sources.py` on other checkpoints | Proven on DeepSeek-V4.1-Flash-1017 IQ2/IQ3 only; the tensor→layer mapping (compressor/indexer placement) is checkpoint-specific |
| Candidate fallback values | Valid for DeepSeek-V4.1-Flash-1017 (source layer 20, top-k 2048, block size 8). Other checkpoints need their own cross-check |

## Engine-side gaps (upstream, not patched here)

- **Unguarded candidate division.** `cand_blk` defaults to 0 (`ds4_model.cu:502-503`)
  and `(n[t] + cand_blk - 1) / cand_blk` (`:12508`) has no zero guard — first forward
  crashes with `0xC0000094` when metadata lacks the candidate keys. The packer fallback
  prevents it, but the engine-level fix (`max(1, cand_blk)` or skipping the candidate
  path when `cand_src == -1`) belongs upstream. Crash evidence: WER offset `0x94778`,
  disassembly shows `idiv ecx` with `ecx = g.cand_blk` (~95% confidence, no PDB symbols).
- **No Q8_0 Engram kernel.** Whitelist `ds4_model.cu:3163-3196`; kernels
  `ds4_kernels.cu:2172/2719` implement only Q6_K / IQ4_NL / FP8. Community quants
  shipping Q8_0 Engram tables therefore always need a side set or a conversion.

## Performance ceiling on the reference machine (measured)

- 2× RTX 4090 48 GB + **128 GB RAM** saturates: expert cache hit tops out ~79–89%, warm
  decode ~19–20 tok/s at 96K ctx. Per-token NVMe random reads ~20 MiB; RAM available
  during decode 0.35–0.64 GiB. The bottleneck is expert-cache capacity, not GPU or
  sequential NVMe bandwidth — but the *magnitude* of a 192 GB RAM upgrade is **not
  derivable** from these samples (3 short runs; engine cache occupancy not exposed).
- Cold start ~15.8 s wall / ~3.9 s TTFT on this hardware.

## Scope limits of the toolkit

- Packer-side only. The Maya DS4 / Strata engine is never modified here.
- No model weights, GGUF shards, or Engram tables are distributed by this repository.
- All benchmarks are single-machine (2× RTX 4090, Windows 11); no third-party
  reproduction has been reported to us yet.
