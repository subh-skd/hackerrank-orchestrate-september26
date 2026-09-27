"""Typed records for every dataset file.

Money is Decimal everywhere. The published samples carry values such as
17229139.2 and 603.3 that must be reproduced exactly, so binary floats are
never used for an amount.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

# Cash-state groupings used throughout the engine. Derived from problem_statement.md
# ("Ignore pending credits, failed or cancelled transactions, duplicate records,
# and unrealized investments") and AGENTS.md 6.1.
SETTLED = "settled"
PENDING = "pending"
SCHEDULED = "scheduled"
CANCELLED = "cancelled"
FAILED = "failed"
UNREALIZED = "unrealized"

DEBIT = "debit"
CREDIT = "credit"
NON_CASH = "non_cash"

FLEX_FIXED = "fixed"
FLEX_STOPPABLE = "stoppable"
FLEX_REDUCIBLE = "reducible"
FLEX_BOTH = "reducible_or_stoppable"


@dataclass(frozen=True)
class Profile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: frozenset[str]
    expense_categories_user_is_willing_to_reduce: frozenset[str]
    expense_categories_user_is_willing_to_stop: frozenset[str]
    payment_methods_user_will_consider: frozenset[str]
    max_installment_months: int | None

    @property
    def considers_full_payment(self) -> bool:
        return "full_payment" in self.payment_methods_user_will_consider

    @property
    def considers_partial_payment(self) -> bool:
        return "partial_payment" in self.payment_methods_user_will_consider

    @property
    def considers_installments(self) -> bool:
        return "installments" in self.payment_methods_user_will_consider

    def may_stop(self, category: str) -> bool:
        """A category may be stopped only if permitted and not protected."""
        if category in self.expense_categories_to_protect:
            return False
        return category in self.expense_categories_user_is_willing_to_stop

    def may_reduce(self, category: str) -> bool:
        if category in self.expense_categories_to_protect:
            return False
        return category in self.expense_categories_user_is_willing_to_reduce


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Decimal | None  # None for the 16 rows whose amount lives in an image
    currency: str
    event_date: date
    settlement_date: date | None
    status: str
    linked_event_id: str | None
    flexibility: str
    minimum_allowed_amount: Decimal | None

    @property
    def cash_date(self) -> date:
        """The date cash actually moves. Settlement wins when present."""
        return self.settlement_date or self.event_date

    @property
    def is_income(self) -> bool:
        return self.direction == CREDIT

    @property
    def is_salary(self) -> bool:
        return self.event_type == "income" and self.category == "salary"

    @property
    def is_stoppable(self) -> bool:
        return self.flexibility in (FLEX_STOPPABLE, FLEX_BOTH)

    @property
    def is_reducible(self) -> bool:
        return self.flexibility in (FLEX_REDUCIBLE, FLEX_BOTH)

    @property
    def is_flexible(self) -> bool:
        return self.flexibility != FLEX_FIXED


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str
    # Present only in sample_requests.csv; never read on the prediction path.
    expected: dict[str, str] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal

    def schedule(self) -> list[tuple[date, Decimal]]:
        """Explicit payment schedule: first_payment_date + k * frequency.

        Verified to the day against solved samples 02, 07 and 12.
        """
        from datetime import timedelta

        step = self.payment_frequency_days or 0
        return [
            (self.first_payment_date + timedelta(days=step * k), self.payment_amount)
            for k in range(self.number_of_payments)
        ]

    @property
    def last_payment_date(self) -> date:
        return self.schedule()[-1][0]


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: str
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None

    @property
    def path(self) -> str:
        return f"media/images/{self.image_id}.png"


@dataclass(frozen=True)
class Rate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal


@dataclass
class Dataset:
    """Everything loaded, indexed for lookup."""

    profiles: dict[str, Profile]
    events_by_user: dict[str, list[Event]]
    events_by_id: dict[str, Event]
    requests: list[Request]
    samples: list[Request]
    options_by_request: dict[str, list[PaymentOption]]
    messages_by_user: dict[str, list[Message]]
    images_by_event: dict[str, ImageRef]
    images: list[ImageRef]
    rates: dict[tuple[date, str, str], Decimal]

    def options(self, request_id: str) -> list[PaymentOption]:
        return self.options_by_request.get(request_id, [])

    def events(self, user_id: str) -> list[Event]:
        return self.events_by_user.get(user_id, [])

    def messages(self, user_id: str) -> list[Message]:
        return self.messages_by_user.get(user_id, [])
