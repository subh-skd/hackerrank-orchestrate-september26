"""CSV readers. Every field is parsed and validated once, here.

Blank amounts are preserved as None (problem_statement.md: "Do not treat a blank
amount as zero") and later resolved from the linked image.
"""

from __future__ import annotations

import csv
import os
from datetime import date
from decimal import Decimal
from pathlib import Path

from .models import (
    Dataset,
    Event,
    ImageRef,
    Message,
    PaymentOption,
    Profile,
    Rate,
    Request,
)

OUTPUT_COLUMNS = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)


def repo_root() -> Path:
    """The repository root, resolved relative to this file rather than the cwd."""
    return Path(__file__).resolve().parents[2]


def dataset_dir() -> Path:
    """dataset/ beside code/, unless BOW_DATASET points elsewhere."""
    override = os.environ.get("BOW_DATASET")
    return Path(override).resolve() if override else repo_root() / "dataset"


def _rows(path: Path) -> list[dict[str, str]]:
    # utf-8-sig tolerates a BOM; the message file contains pre-existing U+FFFD
    # characters from the source data, which decode cleanly and are left as-is.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _dec(raw: str | None) -> Decimal | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    return Decimal(text)


def _req_dec(raw: str) -> Decimal:
    value = _dec(raw)
    if value is None:
        raise ValueError("required amount is blank")
    return value


def _date(raw: str | None) -> date | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    return date.fromisoformat(text)


def _req_date(raw: str) -> date:
    value = _date(raw)
    if value is None:
        raise ValueError("required date is blank")
    return value


def _int(raw: str) -> int | None:
    text = (raw or "").strip()
    return int(text) if text else None


def _pipe(raw: str) -> tuple[str, ...]:
    text = (raw or "").strip()
    if not text:
        return ()
    return tuple(part.strip() for part in text.split("|") if part.strip())


def _bool(raw: str) -> bool:
    return (raw or "").strip().lower() == "true"


def load_profiles(path: Path) -> dict[str, Profile]:
    out: dict[str, Profile] = {}
    for row in _rows(path):
        profile = Profile(
            user_id=row["user_id"],
            home_currency=row["home_currency"],
            current_available_balance=_req_dec(row["current_available_balance"]),
            minimum_balance_to_keep=_req_dec(row["minimum_balance_to_keep"]),
            financial_priorities=_pipe(row["financial_priorities"]),
            expense_categories_to_protect=frozenset(
                _pipe(row["expense_categories_to_protect"])
            ),
            expense_categories_user_is_willing_to_reduce=frozenset(
                _pipe(row["expense_categories_user_is_willing_to_reduce"])
            ),
            expense_categories_user_is_willing_to_stop=frozenset(
                _pipe(row["expense_categories_user_is_willing_to_stop"])
            ),
            payment_methods_user_will_consider=frozenset(
                _pipe(row["payment_methods_user_will_consider"])
            ),
            max_installment_months=_int(row["max_installment_months"]),
        )
        out[profile.user_id] = profile
    return out


def load_events(path: Path) -> list[Event]:
    return [
        Event(
            event_id=row["event_id"],
            user_id=row["user_id"],
            event_type=row["event_type"],
            description=row["description"],
            category=row["category"],
            direction=row["direction"],
            amount=_dec(row["amount"]),
            currency=row["currency"],
            event_date=_req_date(row["event_date"]),
            settlement_date=_date(row["settlement_date"]),
            status=row["status"],
            linked_event_id=(row["linked_event_id"] or "").strip() or None,
            flexibility=row["flexibility"],
            minimum_allowed_amount=_dec(row["minimum_allowed_amount"]),
        )
        for row in _rows(path)
    ]


def load_requests(path: Path, with_expected: bool = False) -> list[Request]:
    out: list[Request] = []
    for row in _rows(path):
        expected = None
        if with_expected:
            expected = {col: (row.get(col) or "") for col in OUTPUT_COLUMNS[1:]}
        out.append(
            Request(
                request_id=row["request_id"],
                user_id=row["user_id"],
                request_date=_req_date(row["request_date"]),
                request_type=row["request_type"],
                requested_amount=_req_dec(row["requested_amount"]),
                desired_completion_date=_req_date(row["desired_completion_date"]),
                allows_partial_payment=_bool(row["allows_partial_payment"]),
                request_text=row["request_text"],
                expected=expected,
            )
        )
    return out


def load_options(path: Path) -> list[PaymentOption]:
    return [
        PaymentOption(
            payment_option_id=row["payment_option_id"],
            request_id=row["request_id"],
            payment_method=row["payment_method"],
            payment_amount=_req_dec(row["payment_amount"]),
            number_of_payments=int(row["number_of_payments"]),
            first_payment_date=_req_date(row["first_payment_date"]),
            payment_frequency_days=_int(row["payment_frequency_days"]),
            financing_fee=_dec(row["financing_fee"]) or Decimal(0),
            total_payable_amount=_req_dec(row["total_payable_amount"]),
        )
        for row in _rows(path)
    ]


def load_messages(path: Path) -> list[Message]:
    return [
        Message(
            message_id=row["message_id"],
            user_id=row["user_id"],
            request_id=(row["request_id"] or "").strip() or None,
            related_event_id=(row["related_event_id"] or "").strip() or None,
            sent_at=row["sent_at"],
            source_type=row["source_type"],
            message_text=row["message_text"],
        )
        for row in _rows(path)
    ]


def load_images(path: Path) -> list[ImageRef]:
    return [
        ImageRef(
            image_id=row["image_id"],
            user_id=row["user_id"],
            request_id=(row["request_id"] or "").strip() or None,
            related_event_id=(row["related_event_id"] or "").strip() or None,
        )
        for row in _rows(path)
    ]


def load_rates(path: Path) -> list[Rate]:
    return [
        Rate(
            rate_date=_req_date(row["rate_date"]),
            from_currency=row["from_currency"],
            to_currency=row["to_currency"],
            rate=_req_dec(row["rate"]),
        )
        for row in _rows(path)
    ]


def load_dataset(base: Path | None = None) -> Dataset:
    root = base or dataset_dir()

    profiles = load_profiles(root / "financial_profiles.csv")
    events = load_events(root / "financial_events.csv")
    requests = load_requests(root / "requests.csv")
    samples = load_requests(root / "sample_requests.csv", with_expected=True)
    options = load_options(root / "request_payment_options.csv")
    messages = load_messages(root / "messages.csv")
    images = load_images(root / "images.csv")
    rates = load_rates(root / "exchange_rates.csv")

    events_by_user: dict[str, list[Event]] = {}
    for event in events:
        events_by_user.setdefault(event.user_id, []).append(event)
    for bucket in events_by_user.values():
        bucket.sort(key=lambda e: (e.cash_date, e.event_id))

    options_by_request: dict[str, list[PaymentOption]] = {}
    for option in options:
        options_by_request.setdefault(option.request_id, []).append(option)
    for bucket in options_by_request.values():
        bucket.sort(key=lambda o: o.payment_option_id)

    messages_by_user: dict[str, list[Message]] = {}
    for message in messages:
        messages_by_user.setdefault(message.user_id, []).append(message)
    for bucket in messages_by_user.values():
        bucket.sort(key=lambda m: (m.sent_at, m.message_id))

    return Dataset(
        profiles=profiles,
        events_by_user=events_by_user,
        events_by_id={e.event_id: e for e in events},
        requests=requests,
        samples=samples,
        options_by_request=options_by_request,
        messages_by_user=messages_by_user,
        images_by_event={i.related_event_id: i for i in images if i.related_event_id},
        images=images,
        rates={(r.rate_date, r.from_currency, r.to_currency): r.rate for r in rates},
    )
