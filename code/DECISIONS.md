# Decision register

Where the problem statement leaves room for interpretation, this file records the readings considered, the evidence used to choose, and what is still uncertain. The evidence is almost always one of the 25 solved samples in `dataset/sample_requests.csv`.

---

## Agent architecture

### A1. The LLM makes the decision; code verifies it
**Choice:** for every request, a LangGraph agent builds a case file and asks the local model to choose a plan by the statement's ranking. The graph is prepare → decide → verify, with one retry and then a fallback. `verify_submission` (`bow/agent/verifier.py`) is the gate. It rejects a choice that breaks an eligibility rule, is unsafe, is outranked by another safe plan, or uses more disruptive spending changes than needed, and it feeds the specific reason back to the model for one retry.
**Why not let code decide?** The brief asks for an AI agent.
**Why not trust the LLM unverified?** The statement explicitly encourages "deterministic verification". A small model can be confidently wrong about a threshold, which would cost exact-match points.
**Fallback:** if the model cannot reach an accepted choice after the retry, the exhaustive plan search answers, and the trace and usage report flag the request as a fallback.
**Tested:** `tests/test_agent_offline.py` replaces the model with a scripted one whose first answer is deliberately wrong. On all 25 samples the verifier rejected it and the retry recovered.

### A2. Message evidence
Messages are normalized into a closed set of 30 typed intents by an exact English/Indonesian template parser (`bow/evidence/rules.py`). It covers 214 of 215 messages; the one it misses carries no numbers. Each intent's effect on the forecast is applied deterministically (`bow/evidence/apply.py`). Instructions embedded in a message have no field to land in. The raw message text is also included in the model's case file as context.

### A3. Why not a hosted LLM
A Gemini agent (a tool-calling planner and a multimodal evidence reader) was built and tested first. The available key was on the free tier, which allows 20 requests per day on `gemini-3.5-flash`, far below the roughly 1,000 calls a full agentic run needs. That implementation was removed in favor of local inference.

### A3b. LLM: local Qwen3-1.7B on CPU through Hugging Face
**Why local:** no API quota, cost, or network dependency. The machine has no discrete GPU (Ryzen 5 5600H, 15 GB RAM).

**Why this model and configuration:** measured on that CPU, using real case files:

| Model | Prompt | Precision | Time / decision | Best plan chosen |
|---|---|---|---:|---:|
| Qwen2.5-0.5B-Instruct | full (1,845 tok) | bf16 | 64 s | poor explanation |
| Qwen2.5-1.5B-Instruct | full (1,845 tok) | bf16 | 148 s | broke the schema |
| Qwen2.5-1.5B-Instruct | compact (~350 tok) | fp32 | 9.6 s | 4 / 5 |
| **Qwen3-1.7B (non-thinking)** | **compact (~350 tok)** | **fp32** | **11.9 s** | **5 / 5** |

- **Instruct build, not a raw base checkpoint.** A base model cannot follow instructions or emit JSON reliably.
- **float32.** bfloat16 has no fast kernels on AVX2 CPUs.
- **Non-thinking mode.** Long reasoning traces are too slow on a CPU.
- **Own wrapper.** `ChatHuggingFace` does not forward `enable_thinking`, so the model is wrapped as a LangChain `BaseChatModel` (`bow/agent/local_llm.py`).

**Division of labor:** the graph simulates every candidate plan up front, and the model applies the ranking and chooses. A small model is reliable at the second task and not at the first.

**Explanations:** the final text is the house-style draft for the plan the model chose, which keeps wording consistent with the solved samples. The model's own one-sentence reason is recorded in each trace.

### A4. Determinism
The model decodes greedily (`do_sample=False`), so the same case file always gets the same answer. On top of that:
- a disk-backed LangChain LLM cache replays model answers, deserializing only an explicit allowlist of output classes;
- a per-request result store, keyed by model and image evidence, lets an interrupted run resume and makes reruns reproduce `output.csv` without new inference.

### A5. The ML model is evaluated honestly
A self-supervised `HistGradientBoostingRegressor` with quantile loss forecasts the next grocery, transport or dining amount. On a time-ordered holdout it beats every statistical baseline at q=0.5, 0.75 and 0.9 (see `evaluation/ml_report.md`). It is offered to the calibration search as `variable_estimator = ml_q50 | ml_q75 | ml_q90`, where it scored 107, 97 and 93 of 150 sample field matches, against 110 for the statistical rule. The answer key's notion of "conservative" is less conservative than an upper quantile. The statistical rule therefore stays selected. The model remains available and documented, but it does not drive the submitted predictions.

---

## Output semantics

### D1. `amount_safe_to_pay` measures capacity, independent of payment preference
**Choice:** the largest payment today that keeps the whole 90-day curve at or above the minimum, before spending changes, capped at the request, and reported even when the recommendation is `wait` or `not_recommended`.
**Evidence:** sample 12 reports the full 65,164 as safe while recommending installments, because that user does not accept full payment. Sample 05 is `not_affordable` yet reports 737.
**Implementation:** a payment on day *p* lowers every later day by the same amount, so capacity is `clamp(trough − minimum, 0, requested)`. It is a closed form, not a search.

### D2. `earliest_date_for_full_payment` ignores spending changes
**Evidence:** samples 06 and 21 pay in full today *with* spending changes, yet report a later earliest date. In sample 11 that date even falls after the deadline.

### D3. The deadline is a hard constraint, not a tie-breaker
**Evidence:** in sample 11, waiting would finish on 2025-07-15, after the 2025-06-12 deadline. The answer is to pay today with a spending change, so a late plan is ineligible rather than merely ranked lower.

### D4. `wait` requires the user to accept `full_payment`
**Evidence:** all six `wait` samples (03, 04, 08, 13, 18, 23) belong to users who accept full payment. All three partial-only users (14, 15, 24) receive `not_recommended`.

### D5. `max_installment_months` is compared with `number_of_payments`
**Alternatives:** compare with the schedule's span in months instead.
**Evidence:** both readings fit all samples. The count reading is simpler and agrees with every profile: `max_installment_months` is non-blank exactly when installments are accepted, with zero exceptions across 275 profiles.
**Residual risk:** low. It would only differ for 28–31-day schedules near the limit.

### D6. Status when spending changes are needed
Full payment today *with* spending changes is `affordable_with_plan`, not `affordable_now` (samples 06, 11, 21).

---

## Plan selection

### D7. Which spending changes: the least disruptive set that works
**Alternatives:** the most relief first; the fewest changes; the order of the user's preference lists.
**Evidence:** sample 21 stops an 11/month backup plan and cuts a 47/month streaming plan to 23.50, which is 34.50 of relief. Reducing shopping by 76.78 alone would also have worked, so "most relief" and "fewest changes" are both contradicted. Samples 06 and 11 are consistent with least disruption.
**Implementation:** combinations of up to three changes on distinct series, ordered by total projected spending removed across the window, then by size, then by event id. The first set that makes the plan safe wins. Reductions go to `minimum_allowed_amount`.

### D8. The cited event is the latest occurrence of the series
**Evidence:** `stop:event_476` is user_06's December "Family streaming plan" row. `reduce_to:event_989` is user_11's most recent dining row.

### D9. `not_recommended` wording
**Evidence:** the "Although X is available today…" wording appears exactly when the user accepts neither full payment nor installments and the request allows partial payment (samples 14 and 24). Every other case uses "None of the available options…" (samples 05, 10, 15, 20, 25). That rule fits 7 of 7.

---

## Forecast

### D10. Three kinds of spending history
Measured across all 275 users:
- groceries, transport and dining recur on a fixed per-user interval (7, 10, 14, 5 or 21 days) with rotating descriptions;
- every other expense category recurs exactly once a month with a stable description.

**Implementation:** monthly commitments are keyed by (category, description) and projected on their day of the month. The three variable categories are aggregated per category.

### D11. The salary category hides stable and variable income
140 of 275 users have more than one series in category `salary`: payroll ("Payroll credit", "Base salary") alongside commissions, bonuses, platform payouts and freelance invoices.
**Choice:** a stable series (constant amounts, or the last two equal) projects monthly. Variable income is not projected when a message says it is pending (bonus, payout or commission not approved) or when an approved invoice supersedes it. Otherwise it projects conservatively on its own cadence.
**Evidence:** sample 09 is a freelancer with no message whose answer is `affordable_now`, so some income must be projected. For user_11, projecting commission would contradict the message.

### D12. A single income row never becomes a recurring stream
**Evidence:** user_05 has four "Payroll credit" rows followed by one "Final employer payroll". Treating that final row as recurring flipped sample 05 from `not_affordable` to `affordable_now`. Income needs at least two occurrences.

### D13. A confirmed scheduled salary sets the level and the anchor
A `scheduled` salary credit counts on its settlement date, and later months repeat at that amount. This is what makes user_01 (a prorated first salary followed by a scheduled full salary) affordable.

### D14. Forecast estimator: identified by search
The free parameters (estimators, lookbacks, placement, stability tests) live in `ForecastRule`. They are chosen by coordinate descent that maximizes exact matches across all output columns on the samples. The result is saved in `code/config/forecast_rule.json` and applied unchanged to every evaluation request.
**Known limitation:** the generator's exact essential-spending estimator was not recovered. `amount_safe_to_pay` matches to the cent on only a few samples, though most are within a few percent. Hand analysis ruled out smooth daily rates (implied rates scatter from 0.72× to 1.78× of the historical average) and simple per-occurrence means. Because status, method, plan and dates depend on thresholds rather than exact cents, those columns score much higher than the amount column.
**Overfitting guard:** each parameter has a direct financial meaning and a handful of values. The search space is small next to the 150 categorical sample constraints it is scored on.

---

## Evidence

### D15. Conflicting salary levels resolve to the safer figure
"Your confirmed base salary is X" and "the remaining confirmed monthly salary is X" restate a level without a dated change.
**Choice:** cap projected income at X without ever raising it.
**Evidence:** for user_11 the message says 38,760,000 but settled history says 23,256,000, and only the history level reproduces sample 11's earliest date. This follows the statement's last conflict rule, the financially safer interpretation. Dated amendments ("increased to X from D") are applied as stated.

### D16. One-cycle salary changes
"Next salary is reduced to X" and "temporary monthly pay is X for the next payroll" change only the next credit. "Regular salary X plus one-time arrears Y" credits X + Y once, then X.

### D17. Other amendments
- Rent increases multiply future projected rent and housing flows.
- Date changes move the next payday and re-anchor later ones.
- "Income ended" removes all projected income.
- "Regular salary resumes on D" starts a confirmed monthly stream at D. The new childcare payment in the same message has no stated amount, so none is invented.

### D18. Message extraction: exact template parser
The 215 messages are drawn from a small set of English and Indonesian templates. An exact parser (`bow/evidence/rules.py`) classifies 214 of them into typed intents with their amounts, dates and percentages; the remaining one carries no numbers. See A2.

### D18b. Reading document images locally: OCR, then the local model, then a grounding check
**Pipeline (`bow/evidence/images_local.py`):**
1. RapidOCR (ONNX runtime, CPU) extracts text boxes.
2. Boxes are regrouped into visual lines, so each label stays beside its figure.
3. Qwen3-1.7B receives those lines together with the ledger row they belong to (description, status, dates) and names the one amount that row represents.
4. The answer is accepted only if that exact number appears in the OCR text.

**Why not a vision-language model:** on this CPU, a 2B VLM would take minutes per image and add a multi-gigabyte download. The user chose OCR plus the local model.

**Results:** all 16 blank amounts resolved and grounded. Checked against evidence:
- `image_02` → 100,000 (Balance Due, not the 2,00,000 Total), which reproduces sample 16.
- `image_05` → 822.05, the amount due after 06-Feb-2026, which applies on the row's 9 Feb settlement date.
- `image_12` → 33.50 (subtotal equals total).
- `image_04` → 2,854.

**Known limitation:** `image_14` is a handwritten pharmacy receipt with a total of ₹4,543 (its line items sum to that). OCR read the handwriting as "S43", and the model returned 43. The grounding check passed because "43" is in the OCR text. The error does not affect any output row, because the event is a one-off settled purchase in user_101's history, and a single occurrence is never projected. A "total must exceed the line items" guard was considered and rejected, since phone and receipt numbers on other documents would trigger false rejections.

### D19. Unresolvable blank amounts are excluded, never zeroed
If an image cannot be read (no file, no key, unparseable result), the event is excluded and flagged in its trace. It is never counted as a zero payment.

---

## Explanations

### D20. Templates, not generation
Wording is mined from the samples and filled with engine-computed figures. The "leaves at least X available" figure is the user's `minimum_balance_to_keep` in every sample that uses it. Dates render as `8 August 2025`, and amounts use thousands separators with two decimals only when fractional. Formatting is verified by `tests/test_fmt.py`: 117 of 117 strings round-trip.
