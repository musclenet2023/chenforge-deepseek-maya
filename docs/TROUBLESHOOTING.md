# Troubleshooting

Symptoms are quoted from real engine logs on Windows; the fix is always packer-side
(re-run the pack after applying the patch), never engine-side.

## 1. `layer 2: compress ratio 2 without a KV source of that ratio before it`

```
[maya] layer split across GPUs [0, 1] (auto)
maya-ds4: layer 2: compress ratio 2 without a KV source of that ratio before it
RuntimeError: the engine exited before it was ready
```

**Cause** — the quant's GGUF lacks `deepseek41.attention.kv_source_layers` and
`deepseek41.attention.indexer.source_layers`, so `pack/ds4.txt` has empty
`kv_source_layer_ids` / `index_source_layer_ids` lines.

**Fix** — use the patched packer (fix 1) and re-pack. Verify with:

```
python tools\verify_kv_mapping.py <pack-dir>
```

Expected output ends with `ALL LAYERS PASS`.

## 2. First forward crashes, exit code `0xC0000094` (STATUS_INTEGER_DIVIDE_BY_ZERO)

Server starts and loads weights fine, then dies on the first generation request.

**Cause** — missing candidate keys; the engine defaults `candidate_block_size = 0`
(`ds4_model.cu:502-503`) and divides by it.

**Fix** — re-pack with the patched packer (fix 2). The packer prints
`candidate geometry not in the GGUF metadata, taken from the IQ2_XXS pack of the same model`
when it applies the fallback. Confirm `pack/ds4.txt` contains:

```
candidate_source_layer_id 20
candidate_topk_blocks 2048
candidate_block_size 8
```

## 3. Packer crashes inside `write_engram` (TypeError on `struct.pack`)

**Cause** — the quant spells the Engram head dim `engram.key_length` (not `engram.head_dim`)
and the pad key `engram.pad_id`; the alias table in the stock packer lacks both.

**Fix** — patched packer (fix 3). No action beyond re-running the pack.

## 4. `Engram RAM: unsupported table type` / engine rejects Q8_0 Engram

**Cause** — the engine's Engram row whitelist is Q6_K / IQ4_NL / MXFP8 / FP8 only
(`ds4_model.cu:3163-3196`); the IQ3_XXS quant's tables are Q8_0.

**Fix** — pass a side IQ4_NL Engram set via `--engram` (fix 4 drops the shards' own Q8_0
tables automatically). For the IQ2_XXS 6-shard layout, the shared
`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` **is** the IQ4_NL set: hardlink it as shard 6
(examples\example_pack_iq2.bat does this) and pass the model folder as `--engram`.

## 5. VRAM check fails at startup / server says BLOCKED

The engine needs the full VRAM budget (both 4090s ~48 GB each in the reference setup).
If another model instance holds VRAM, the startup script reports BLOCKED before touching
anything. Stop the other instance yourself; this repository's scripts never stop processes.

## 6. 192K context

The KV budget at 192K with q8_0 KV is ~8 GB and fits in 96 GB VRAM on paper, but stability
was **not** verified on the reference machine; 96K was verified. Start at 96K.

## 7. Decode is slow (~8 tok/s) on first requests

This is the expert cache warming up. Watch the cache hit statistic climb (59% → 79% in the
benchmark) and decode rise with it (7.9 → 19.1 tok/s). Sustained ~20 tok/s with 128 GB RAM is
the cache-capacity plateau, not a bug — see docs/BENCHMARK.md.
