# IQ3_XXS Compatibility Journey — reusing the method, fixing what's new

The prometheusAIR IQ3_XXS 10-shard set (417.07 GB, Engram Q8_0 in-shard) was the second
quant packed for Maya DS4. It **reused the IQ2_XXS method unchanged** (pack-only, zero-copy,
verify-before-run) and needed **four additional fixes** for its different metadata layout.
Each fix below lists the original error, the root cause with engine-source evidence, the
diff-level change, the verification result, and what remains unverified.

## Fix 1 — KV-source fallback (packer)

| | |
|---|---|
| **Original error** | `maya-ds4: layer 2: compress ratio 2 without a KV source of that ratio before it` → `RuntimeError: the engine exited before it was ready` |
| **Root cause** | The engine requires every `ratio > 0` layer to be preceded by a KV source of the same ratio (`ds4_model.cu:2671`). The IQ3 GGUF omits `deepseek41.attention.kv_source_layers` and `attention.indexer.source_layers` entirely (header dump `tools_phase4/meta_dump.txt`), so the packer wrote empty lines into `ds4.txt`. |
| **Evidence it's metadata-only** | Both quants' `compress_ratios` are identical (0,0,2×18,1×20,0,0,0) and the compressor tensors (`blk.N.attn_compressor_*`) sit exactly on layers 2/8/14/20, indexer tensors on 2/8/14/20/24/28/32/36 — the same values IQ2 carries in metadata. Derived independently from IQ3's own tensors (`derive_kv_sources.py`, MATCH), not copied from IQ2. |
| **Fix** | `geometry()` fallback: derive both lists from the shards' own tensor manifest; `SystemExit` when neither metadata nor tensors can answer. |
| **Before/after** | `ds4.txt` diff is exactly two lines: `kv_source_layer_ids 2,8,14,20` and `index_source_layer_ids 2,8,14,20,24,28,32,36` (backup `ds4.txt.orig`). |
| **Verification** | `verify_kv_mapping.py` → **ALL LAYERS PASS** (all 40 layers + MTP). Engine then started and served — see benchmark below. |
| **Portable?** | Yes — works for any deepseek41 quant that has the tensors but lacks the keys. |

## Fix 2 — Candidate-geometry fallback (packer)

| | |
|---|---|
| **Original error** | First forward pass crashed: exit code `0xC0000094` (STATUS_INTEGER_DIVIDE_BY_ZERO), module `maya-ds4.exe`, faulting offset `0x94778` (WER capture). |
| **Root cause** | `cand_blk = p.num("candidate_block_size", 0)` (`ds4_model.cu:502-503`) defaults to 0 when the GGUF omits the three candidate keys; the window planner then runs `(n[t] + cand_blk - 1) / cand_blk` (`ds4_model.cu:12508`). Disassembly at `0x94778` shows `idiv ecx` with `ecx = g.cand_blk` — confidence ~95% (offsets and source align; no PDB symbols available). |
| **Fix (option A, engine untouched)** | Packer fallback writes `candidate_source_layer_id 20`, `candidate_topk_blocks 2048`, `candidate_block_size 8` — values cross-checked against the IQ2_XXS quant of the same checkpoint (its GGUF carries them natively), which phase-4b had already shown shares the architecture. |
| **Before/after** | `ds4.txt` gains exactly three lines; tensor stats unchanged (48,531 tensors / 12 files). |
| **Verification** | Engine reached Ready and served requests at 96K ctx (see below); no repeat of 0xC0000094. |
| **Portable?** | Yes, per-checkpoint (values are architecture constants, not quantization-dependent). |
| **Unverified residual** | The engine-side divide itself is still unguarded upstream — a defensive `max(1, cand_blk)` engine patch is drafted but *not applied* (would require recompiling the engine; out of scope). |

## Fix 3 — Engram key aliases (`engram.key_length`, `engram.pad_id`)

| | |
|---|---|
| **Original error** | Packer crash in `write_engram()`: `TypeError: required argument is not an integer` — reproducible in a header-only dry-run. |
| **Root cause** | prometheus spells the Engram head dim `deepseek41.engram.key_length` (=256) and the pad token `engram.pad_id`; the alias table only knew `engram_head_dim`/`engram.pad_token_id` (underlined forms). The dry-run proved the failure is a hard crash at `struct.pack("<4s6i", …, hd, …)`, occurring *after* `tensors.txt` is written — no corrupt output is possible. |
| **Fix** | Two alias rows in `NEWER_KEYS`: `("engram_key_length", "engram.head_dim")`, `("engram.key_length", "engram.head_dim")` and `("engram.pad_id", "engram.pad_token_id")` (the dotted spellings matter — the metadata keys are dotted). |
| **Verification** | After aliasing, `engram.head_dim = 256`; DSE1 header `("DSE1",129280,2,4,8,256,2)` packs successfully. |
| **Portable?** | Yes — trivially, for any quant using either spelling. |

## Fix 4 — Side-Engram with in-shard Q8_0 tables

| | |
|---|---|
| **Problem** | The IQ3 shards carry their Engram tables **in-shard as Q8_0** (`blk.1/blk.14.engram_embd.weight`, [256, 384,006,168], 208.9 GB). The engine has **no Q8_0 Engram kernel** — only Q6_K / IQ4_NL / MXFP8 / FP8 (`ds4_model.cu:3163-3196`, decode kernels `ds4_kernels.cu:2172/2719`). It would load, then reject the table. |
| **Fix** | Two packer changes: (a) fill `engram.num_embeddings` from the shards' own per-layer table dims (prometheus lacks this KV, so `split_ple()` had nothing to split on); (b) when `--engram` is given, **drop** the shards' own `engram_embd` tensors and take the tables from the side IQ4_NL set (the same 110.595 GB file IQ2 uses — hardlinked, zero copy). Projections (`engram_wkv/q/k`) still come from IQ3's own shards. |
| **Verification** | Precision chain validated by sampling: 64 rows Q8_0-dequant vs IQ4_NL-decode gave MSE 4.39e-3 / **SNR 22.4 dB** — the two files are the same f32 table quantized two ways (hash parameters identical, `engram.source` commit matches). A self-made requant round-trip scored SNR 17.3 dB, confirming the decode chain is understood correctly. |
| **Portable?** | Yes, whenever a compatible side Engram set exists for the checkpoint. |
| **Unverified residual** | Q8_0→IQ4_NL requantization was *designed* (streaming, <1 GB RAM, ~1-3 h) and its algorithm validated on samples, but never run to a full 110 GB output — it was unnecessary because the ready-made IQ4_NL file exists. |

## End-to-end result (measured, 2026-11-22)

Server at `127.0.0.1:8094`, ctx 98,304, 2× RTX 4090 48 GB + 128 GB RAM, thinking budget
32,768: **cold 15.84 s / TTFT 3.93 s / 7.9 tok/s; warm 5.99 s / 0.77 s / 18.7 tok/s;
5.84 s / 0.48 s / 19.1 tok/s**; expert cache hit 59.1% → 72.4% → 79.1%. Bottleneck
analysis (RAM-bound expert cache, ~20 MiB NVMe random reads per token) in
`docs/KNOWN_LIMITATIONS.md`. Full methodology: `docs/BENCHMARK.md`.

## Attribution summary

- Fixed **first on IQ2_XXS**: pack-only method, zero-copy/hardlink packing, Engram
  side-pack contract, verification tool, benchmark methodology.
- Fixed **on IQ3_XXS** (new work): KV-source fallback, candidate-geometry fallback,
  Engram key aliases, side-Engram num_embeddings/drop-Q8_0 logic.
- IQ2's GGUF never exercised the fallback paths; the four patch items are IQ3-driven
  additions to a pipeline IQ2 established.
