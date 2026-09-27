"""Buy or Wait? - AI financial decision agent.

Package layout:
  agent/     LangGraph agent: local Qwen model (Hugging Face transformers, CPU),
             verification gate, resumable runner
  evidence/  typed message amendments, OCR-based image reading
  ml/        learned quantile forecast of essential spending (evaluated option)
  the rest   ledger reconstruction, recurrence, 90-day forecast, plan search,
             explanations, contract validator, traces, usage report
"""
