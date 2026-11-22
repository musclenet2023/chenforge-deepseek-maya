# Benchmark

> All numbers are from the author's machine, single configuration, small sample size.
> Treat them as one data point, not a promise. Re-run on your own hardware.

## Reference machine

- CPU: AMD Ryzen 9 (9900X / 9800X3D class)
- GPU: 2× NVIDIA RTX 4090 48 GB (engine splits layers across both)
- RAM: 128 GB — this is the binding constraint, see below
- OS: Windows 11, model weights on local NVMe

## IQ3_XXS @ 96K context, 128 max tokens, single-prompt (42 prompt tokens)

Server: Maya DS4 `llama-server`-style API, `temperature=0.7`, no tools, 3 sequential requests,
same prompt. Streaming measured client-side and server-side.

| Run | Wall | TTFT | Decode tok/s (server) | Expert cache hit | NVMe read |
|---|---|---|---|---|---|
| 1 (cold) | 15.8 s | 3.9 s | 7.9 | 59.1% | 12.9 GiB |
| 2 (warm) | 6.0 s | 0.77 s | 18.7 | 72.4% | 2.0 GiB |
| 3 (warm) | 5.8 s | 0.48 s | 19.1 | 79.1% | 2.1 GiB |

### Interpretation

- The bottleneck is **expert-cache capacity, not GPU compute**: decode speed, cache hit rate
  and NVMe read volume move together across runs. Each warm token draws ~20 MiB of expert
  weights from NVMe at ~340-365 MiB/s effective throughput — random-read latency × miss count,
  not sequential bandwidth, is the limiter.
- With 128 GB RAM ~100% occupied (weights resident + OS), the expert cache saturates around
  79-89% hit rate and decode plateaus near ~20 tok/s. More RAM ⇒ larger expert cache ⇒ fewer
  misses ⇒ faster decode, but the magnitude cannot be projected from this sample.
- GPU utilization peaks of 96-100% (2-second sampling) reflect waiting on weights, not
  compute saturation.

## IQ2_XXS

Packed and load-tested with the same fix set; formal benchmark pending (see docs/TROUBLESHOOTING.md for known issues).

## How to reproduce

1. Pack and verify per docs/INSTALL.md.
2. Start the engine with your chosen context (96K verified on the reference machine; 192K
   fits the KV budget but was not proven stable — see the source project's notes).
3. Run your benchmark client against the OpenAI-compatible endpoint; record TTFT, decode
   tok/s, and the engine's expert-cache hit statistic.
