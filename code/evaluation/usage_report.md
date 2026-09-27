# Token usage and cost report

Generated 2026-09-13 04:34 UTC by the full-dataset run that wrote `output.csv`.

## How the model is used

Each request runs a LangGraph agent whose decision step is a **local** Qwen model downloaded from the Hugging Face Hub and executed on CPU with `transformers` (no API, no GPU). The model receives a compact case file in which every candidate plan has been simulated against the 90-day forecast, and it chooses a plan by the problem statement's ranking. A verifier rejects unsafe, ineligible or outranked choices and returns the reason for one retry. Before the requests run, the same model reads the 16 document images from their OCR text.

## Overall

| Metric | Value |
|---|---:|
| Provider | Hugging Face Hub weights, local CPU inference (`transformers`, LangChain `BaseChatModel`) |
| Models | `Qwen/Qwen3-1.7B` |
| Requests | 250 |
| Decisions accepted from the model | 248 / 250 |
| Model calls (decisions + image readings) | 269 (253 + 16) |
| Decision calls per request (avg / max) | 1.01 / 2 |
| Input tokens | 97,282 |
| Output tokens | 7,674 |
| Total tokens | 104,956 |
| Average tokens per request | 420 |
| Estimated total cost (USD) | $0.00 (local inference, no API fees) |
| Estimated cost per request (USD) | $0.00 |
| CPU inference time, decisions (total / per request) | 58.1 min / 13.9 s |

## Per model

| Model | Calls | Input tokens | Output tokens | Cost (USD) |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3-1.7B` | 269 | 97,282 | 7,674 | $0.00 |
| **Overall** | **269** | **97,282** | **7,674** | **$0.00** |

## Method

Token counts are exact: they come from the model's own tokenizer, attached to every response as LangChain `usage_metadata` and collected by a per-request callback. Image-reading calls are counted from their cached records. There is no per-token price for local inference. The cost is local compute, reported above as CPU time on a 6-core laptop CPU (AMD Ryzen 5 5600H).

Model answers are cached (`code/cache/llm`, `code/cache/results_local`, `code/cache/image_amounts.json`), so re-running reproduces `output.csv` without new inference. The figures above are the calls that produced it.
