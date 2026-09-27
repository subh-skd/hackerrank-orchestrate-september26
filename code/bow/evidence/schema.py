"""The typed evidence contract.

Messages are untrusted. Reading one can only produce an Amendment drawn from
this closed set of intents.
Embedded instructions ("pay the release charge today") have no field to land in,
so they cannot alter policy; the deterministic forecast decides what each
intent does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

# Intents that change the forecast.
SALARY_DATE_CHANGE = "salary_date_change"
SALARY_INCREASE = "salary_increase"
SALARY_NEXT_REDUCED = "salary_next_reduced"
SALARY_TEMPORARY = "salary_temporary"
FIRST_SALARY = "first_salary"
SALARY_CONFIRMED_FX = "salary_confirmed_fx"
SALARY_PLUS_ARREARS = "salary_plus_arrears"
SALARY_RESUMES = "salary_resumes"
INVOICE_APPROVED = "invoice_approved"
BASE_SALARY_CONFIRMED = "base_salary_confirmed"
HOUSEHOLD_INCOME_REMAINING = "household_income_remaining"
INCOME_ENDED = "income_ended"
RENT_INCREASE = "rent_increase"
FAILED_DEBIT_RETRY = "failed_debit_retry"
INTERNAL_TRANSFER = "internal_transfer"

# Intents that confirm something must NOT be counted, or change nothing.
BONUS_PENDING = "bonus_pending"
GIG_PAYOUT_PENDING = "gig_payout_pending"
PRIZE_PENDING = "prize_pending"
PRIZE_SETTLED = "prize_settled"
PORTFOLIO_VALUE_CHANGE = "portfolio_value_change"
INVESTMENT_SALE_SETTLED = "investment_sale_settled"
REFUND_PENDING = "refund_pending"
FX_REFUND_PROCESSING = "fx_refund_processing"
FX_CHARGE_PENDING = "fx_charge_pending"
DISPUTE_OPEN = "dispute_open"
REIMBURSEMENT_CLOSED = "reimbursement_closed"
SEPARATE_CARD_MINIMUMS = "separate_card_minimums"
RECEIPT_CONFIRMATION = "receipt_confirmation"
SCAM_OR_SOLICITATION = "scam_or_solicitation"
NO_EFFECT = "no_effect"

FORECAST_INTENTS = frozenset(
    {
        SALARY_DATE_CHANGE,
        SALARY_INCREASE,
        SALARY_NEXT_REDUCED,
        SALARY_TEMPORARY,
        FIRST_SALARY,
        SALARY_CONFIRMED_FX,
        SALARY_PLUS_ARREARS,
        SALARY_RESUMES,
        INVOICE_APPROVED,
        BASE_SALARY_CONFIRMED,
        HOUSEHOLD_INCOME_REMAINING,
        INCOME_ENDED,
        RENT_INCREASE,
        FAILED_DEBIT_RETRY,
        INTERNAL_TRANSFER,
    }
)

ALL_INTENTS = tuple(
    sorted(
        FORECAST_INTENTS
        | {
            BONUS_PENDING,
            GIG_PAYOUT_PENDING,
            PRIZE_PENDING,
            PRIZE_SETTLED,
            PORTFOLIO_VALUE_CHANGE,
            INVESTMENT_SALE_SETTLED,
            REFUND_PENDING,
            FX_REFUND_PROCESSING,
            FX_CHARGE_PENDING,
            DISPUTE_OPEN,
            REIMBURSEMENT_CLOSED,
            SEPARATE_CARD_MINIMUMS,
            RECEIPT_CONFIRMATION,
            SCAM_OR_SOLICITATION,
            NO_EFFECT,
        }
    )
)


@dataclass(frozen=True)
class Amendment:
    """One normalized fact extracted from one message."""

    message_id: str
    user_id: str
    intent: str
    amount: Decimal | None = None
    currency: str | None = None
    secondary_amount: Decimal | None = None  # e.g. one-time arrears
    effective_date: date | None = None
    percent: Decimal | None = None
    related_event_id: str | None = None
    source: str = "rules"

    @property
    def affects_forecast(self) -> bool:
        return self.intent in FORECAST_INTENTS

    def describe(self) -> str:
        parts = [self.intent]
        if self.amount is not None:
            parts.append(f"{self.currency or ''} {self.amount}".strip())
        if self.secondary_amount is not None:
            parts.append(f"+{self.secondary_amount}")
        if self.percent is not None:
            parts.append(f"{self.percent}%")
        if self.effective_date:
            parts.append(f"from {self.effective_date.isoformat()}")
        return " ".join(parts)
