# Hugging Face Publishing Plan — DRAFT, nothing created or uploaded

Status: **plan only**. No HF repository exists, no file uploaded, no token used.
`hf auth whoami` → not logged in (2026-11-23); a token from you is required before
anything in Phase C can proceed, and even then only after your approval.

## 1. Repository type decision

**Dataset Repository** — recommended.

Name suggestion: `ChenForge/deepseek-v4.1-flash-maya-ds4-compatibility`
(substitute your HF username for "ChenForge"; it must not impersonate DeepSeek or Maya).

Rationale: the content is compatibility data, verification records, metadata examples
and tools — none of it is a loadable model, so a Model Repository would misrepresent it
(and is explicitly out unless real weights are ever approved, see §5).

## 2. Proposed file list (dataset repo)

| File | Source | Contains | License |
|---|---|---|---|
| README.md (HF format, YAML frontmatter below) | new | matrix, status, links | MIT |
| compatibility-matrix.md | docs content | IQ2/IQ3 matrix, verification status | MIT |
| pack-ds4-txt-examples/ | local pack dirs | `ds4.txt` (IQ2 + IQ3), tiny (<2 KB) | MIT |
| engram-dse1-header-spec.md | docs/ENGRAM_INTEGRATION.md | DSE1 layout, aliases | MIT |
| tools/pack_gguf_patched.py, verify_kv_mapping.py, derive_kv_sources.py | ChenForge repo | tools (same as GitHub) | MIT |
| verify-kv-log-iq3.txt | local run log | ALL LAYERS PASS output | MIT |
| benchmark-iq3-96k.md | docs/BENCHMARK.md | measured numbers + methodology | MIT |
| SOURCES.md | new | upstream model links + SHA256s we verified (IQ4_NL Engram `3dac4c54…9fb4f3`, etc.) | MIT |

No internal paths, no account names, no credentials in any file (same rules as GitHub).

YAML frontmatter (draft):

```yaml
license: mit
tags:
- deepseek
- deepseek-v4.1-flash
- gguf
- quantization
- compatibility
- maya-ds4
- windows
pretty_name: "ChenForge — DeepSeek V4.1 Flash Compatibility Toolkit"
```

## 3. Explicitly NOT uploaded (independent assessment)

| Content | Assessment |
|---|---|
| Original/modified GGUF shards | **Do not publish.** Weights are third-party redistributions (ISTA-DASLab GSQ-RCO quant; prometheusAIR quant); re-hosting 250–417 GB adds no value over the originals, duplicates storage, and raises license/attribution obligations. Users should download from the original repos. |
| Engram side-pack (IQ4_NL 110.6 GB) | **Do not publish.** It is ISTA-DASLab's file byte-for-byte (we hardlink it); the upstream card already says "not re-hosted here". Re-hosting would duplicate their distribution. |
| Maya-specific large pack | **Do not publish.** The pack is metadata-only (~9.5 MB) — valuable parts (`ds4.txt`, `engram.bin` header) go in as small examples instead; the full pack depends on local shard layout and would confuse users. |
| Any weights | See §5 gate. |

## 4. What the dataset repo gives the community

- A HF-discoverable mirror of the compatibility matrix and error map (GitHub is where
  the code lives; HF is where quant users look first).
- Machine-checkable `ds4.txt` examples so users can diff their own pack against a
  known-good geometry.
- SHA256 pins so users can verify they have the exact files the fixes were proven on.

## 5. Gate for ever publishing weights (if ever proposed)

1. Written redistribution basis from upstream licenses (ISTA-DASLab GGUFs are
   MIT-labeled; DeepSeek base is MIT; GSQ-RCO quantization provenance must be quoted).
2. Attribution: DeepSeek AI (base), ISTA-DASLab (quant + Engram), AtomicChat
   (uncensored edit), prometheusAIR (IQ3 quant), Niko1221/Strata (engine).
3. Size & capacity: 250–417 GB per quant; free HF storage limits.
4. SHA256 for every file, published in SOURCES.md.
5. Load instructions tied to a specific ChenForge release.
6. Added value vs existing repos — currently **none**; the existing repos already
   serve these files. Default answer: no.

## 6. Steps once you approve

1. You provide a HF token (fine-grained, write scope on one dataset repo) — entered via
   `hf auth login` yourself or a vault prompt; it is never displayed or stored.
2. Confirm the HF username and repo name.
3. Create the dataset repo (private first, then public after you inspect it).
4. Upload the §2 list; verify each file resolves.
5. Cross-link: ChenForge README adds the HF link; HF README links back to GitHub.
