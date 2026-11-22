# Install

## Prerequisites

| Component | Requirement |
|---|---|
| OS | Windows 10/11 x64 |
| Python | 3.10 or newer |
| Engine | Maya DS4 / Strata runtime with `deepseek41` support (built separately; see the Strata project) |
| GPU | NVIDIA CUDA GPU(s); the reference setup is 2× RTX 4090 48 GB |
| System RAM | 128 GB minimum for IQ3_XXS (see BENCHMARK.md — expert cache is RAM-bound) |
| Python packages | `huggingface_hub` (for `tools/download_hf_model.py`) |

## Disk requirements

| Quant | Source size | Output (pack metadata + IQ4_NL Engram) | Peak during pack |
|---|---|---|---|
| IQ2_XXS 6-shard | ~139 GB backbone + 110.6 GB Engram | metadata only (~10 MB) | +0 (hardlink shard 6) |
| IQ3_XXS 10-shard | ~417 GB | ~352 GB (incl. re-quantized IQ4_NL Engram) | ~770 GB free required |

The packer writes only pack metadata (`ds4.txt`, `tensors.txt`, `engram.bin`, `tokenizer/`)
beside the weights — no weight data is copied — **except** the Engram re-quantization step
(Q8_0 → IQ4_NL, ~2-5 hours for the IQ3_XXS tables).

## Steps

1. **Get the weights** (not from this repository):

   ```
   python tools\download_hf_model.py <hf-repo-id> D:\models\<folder>
   ```

   For the IQ2_XXS 6-shard layout, the script hardlinks the Engram file as shard 6
   (equivalent to `mklink /H` in examples\example_pack_iq2.bat).

2. **Pack** — edit `MODEL_DIR` / `ENGRAM_DIR` in one of `examples\example_pack_iq*.bat`,
   then run it. It calls `tools\pack_gguf_patched.py` and then verifies the geometry.

3. **Verify the geometry**:

   ```
   python tools\verify_kv_mapping.py D:\models\<folder>\pack
   ```

   Expected: `ALL LAYERS PASS: every ratio>0 layer has a KV source of the same ratio before it`
   with `kv_source=[2,8,14,20]`, `index_source=[2,8,14,20,24,28,32,36]`, `engram=[1,14]`,
   `candidate=20`.

4. **Run the engine** pointing at the pack directory (`pack/ds4.txt`). The engine reads the
   weights in place from `MODEL_DIR`; see docs/TROUBLESHOOTING.md if startup fails.

## Optional: derive the KV sources yourself (no trust needed)

```
python tools\derive_kv_sources.py D:\models\<folder>
```

Header-only scan of the GGUF shards; prints the compressor/indexer layers and compares them
with the IQ2_XXS quant's metadata values. No weight data is read.
