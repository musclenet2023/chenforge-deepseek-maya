# ChenForge — DeepSeek V4.1 Flash 相容性工具組

以最小、可重現的「打包器端」修復，讓 Maya DS4 引擎（`deepseek41` 架構執行時，MIT 授權）
能載入缺少必要 metadata 的社群量化版
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)。

> **本倉庫是什麼**：對 `tools/ds4/pack_gguf.py` 的四組小型修復（只動打包器、不動引擎）、
> 驗證工具與 Windows 範例腳本。
> **本倉庫不是什麼**：不含任何模型權重、GGUF 檔案或引擎二進位檔。

## 問題

社群 GGUF 量化版（例如 prometheusAIR 的 **IQ3_XXS**）的 GGUF header 缺少數個 `deepseek41.*`
metadata 鍵，引擎的 geometry 載入器隨即在啟動或第一個 forward 崩潰：

| 缺少的鍵 | 症狀 | 根本原因（引擎原始碼） |
|---|---|---|
| `attention.kv_source_layers`、`attention.indexer.source_layers` | `layer 2: compress ratio 2 without a KV source of that ratio before it` | 任何 `ratio > 0` 的層之前必須存在同 ratio 的 KV source（`ds4_model.cu:2671`） |
| `attention.indexer.candidate_source_layer / candidate_top_k_blocks / candidate_block_size` | 第一個 forward 崩潰，exit `0xC0000094`（整數除以零） | 引擎預設 `cand_blk = 0`（`ds4_model.cu:502-503`） |
| `engram.head_dim` 被寫成 `engram.key_length`、`engram.pad_id` | `write_engram()` 打包 `None`，打包器崩潰 | alias 表缺這兩種拼法 |
| `engram.num_embeddings`（逐層 `blk.N.engram_embd.weight` 佈局） | `split_ple()` 無法切分外部 Engram 集 | 僅憑 metadata 無法取得 dims |

另外，IQ3_XXS 的 Engram 表是 **Q8_0**——引擎沒有 Q8_0 的 Engram kernel（只收
Q6_K / IQ4_NL / MXFP8 / FP8，`ds4_model.cu:3163-3196`）——所以必須用 `--engram` 餵入
**IQ4_NL** 的外部表，並剔除 shards 自帶的 Q8_0 表。

## 修復（只動打包器）

`patches/pack_gguf_ds41-compat.patch.md` 有完整標註的最小 diff；`tools/pack_gguf_patched.py`
是修復後的完整副本。摘要：

1. **KV source fallback** — metadata 缺 `kv_source_layer_ids` / `index_source_layer_ids` 時，
   從 shards 自己的 tensor 清單推導（compressor tensors ⇒ ratio-2 sources，層 2/8/14/20；
   indexer tensors ⇒ ratio-1 sources，層 2/8/14/20/24/28/32/36）。metadata 與 tensors 都無法
   提供答案時明確 `SystemExit`，絕不靜默帶過。
2. **Candidate geometry fallback** — 缺 candidate 三鍵時，取該架構自身的值
   （source layer 20、top-k blocks 2048、block size 8），與同一 checkpoint 的 IQ2_XXS 量化交叉驗證。
3. **Engram 鍵 alias** — `engram.key_length` 視為 `engram.head_dim`、`engram.pad_id` 視為
   `engram.pad_token_id`。
4. **外部 Engram 支援** — 從 shards 的逐層表 dims 補 `engram.num_embeddings`；帶 `--engram`
   時剔除 shards 自帶的 Q8_0 Engram 表，改讀 IQ4_NL 外部集。

## 倉庫結構

```
patches/    pack_gguf_ds41-compat.patch.md   最小 diff（含 file:line 證據標註）
tools/      pack_gguf_patched.py             修復後的打包器完整副本
            verify_kv_mapping.py             對產出的 ds4.txt 重驗引擎的 KV 不變條件
            derive_kv_sources.py             只掃 header：從 GGUF shards 推導 KV/index source
            download_hf_model.py             下載 + SHA256 驗證 + shard-6 硬連結小工具
examples/   example_pack_iq3.bat             Windows 打包範例（IQ3_XXS）
            example_pack_iq2.bat             Windows 打包範例（IQ2_XXS 六 shard）
docs/       INSTALL.md  BENCHMARK.md  TROUBLESHOOTING.md
LICENSE     MIT（本倉庫的修復集）
```

## 快速開始（Windows）

```bat
git clone <本倉庫>
cd deepseek-v4-flash-maya-ds4-windows

:: 1. 下載量化模型（範例：IQ2_XXS uncensored 集）並驗證 SHA256
python tools\download_hf_model.py Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF D:\models\DS41-IQ2XXS

:: 2. 為 Maya DS4 引擎打包（不複製權重，只寫 metadata）
examples\example_pack_iq2.bat        :: 先改 MODEL_DIR

:: 3. 驗證引擎將讀到的 geometry
python tools\verify_kv_mapping.py D:\models\DS41-IQ2XXS\pack
```

硬體需求與磁碟空間見 docs/INSTALL.md。

## 狀態

- IQ2_XXS（六 shard，Abiray uncensored GSQ-RCO）：已打包、geometry 已驗證、server 載入已測試。
- IQ3_XXS（十 shard，prometheusAIR）：已打包、geometry 已驗證（ALL LAYERS PASS）、
  已在 2× RTX 4090 + 128 GB RAM、96K context 下完成 server 啟動與 128-token 生成測試。
- 效能數據：docs/BENCHMARK.md；已知故障與修法：docs/TROUBLESHOOTING.md。

## 來源與授權

- **DeepSeek AI** — DeepSeek-V4.1 架構與基礎權重（MIT，上游 `LICENSE.deepseek-v41`）。
- **ISTA-DASLab** — GSQ-RCO 量化管線與共享的 `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`
  Engram 表（IQ2_XXS GGUF 集標記 MIT）。
- **Project Maya / Niko1221（Strata）** — DS4 引擎與原始 `tools/ds4/pack_gguf.py`（MIT）。
- **prometheusAIR** — IQ3_XXS 量化；其 metadata 缺鍵促成了修復 1-4。
- 本倉庫自身的修復集：MIT。模型權重沿用上游授權，**不**隨本倉庫轉散。
