"""Deterministic message parser for the closed English/Indonesian template set.

Classifies 214 of the 215 messages into typed intents with their amounts,
dates and percentages (the remaining one carries no numbers).

Patterns are ordered: the first match wins, and more specific intents come
before general ones (e.g. "base salary ... commission pending" before
"commission pending").
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

from ..models import Message
from . import schema as s

_CUR = r"(ZAR|IDR|EUR|INR|USD)"
_AMT = r"([\d][\d,]*(?:\.\d+)?)"
_DATE = r"(\d{4}-\d{2}-\d{2})"


def _amount(raw: str) -> Decimal:
    return Decimal(raw.replace(",", ""))


def _date(raw: str) -> date:
    return date.fromisoformat(raw)


# (intent, compiled pattern, extractor). Extractor maps a match to fields.
_Pattern = tuple[str, "re.Pattern[str]", object]


def _p(intent: str, pattern: str, fields=None) -> _Pattern:
    return (intent, re.compile(pattern, re.IGNORECASE | re.DOTALL), fields)


def _cur_amt(m: re.Match, date_group: int | None = None) -> dict:
    out = {"currency": m.group(1), "amount": _amount(m.group(2))}
    if date_group is not None:
        out["effective_date"] = _date(m.group(date_group))
    return out


PATTERNS: list[_Pattern] = [
    # --- solicitation / scam: never a financial fact ------------------------
    _p(
        s.SCAM_OR_SOLICITATION,
        r"(release charge|processing charge|biaya pencairan|biaya pemrosesan)",
    ),
    # --- income -------------------------------------------------------------
    _p(
        s.SALARY_DATE_CHANGE,
        r"(?:confirmed salary is now expected on|gaji yang sudah dikonfirmasi kini diperkirakan masuk pada) "
        + _DATE,
        lambda m: {"effective_date": _date(m.group(1))},
    ),
    _p(
        s.SALARY_INCREASE,
        r"(?:monthly salary has increased to|gaji bulanan anda naik menjadi) "
        + _CUR
        + r" "
        + _AMT
        + r".*?(?:applies from|berlaku mulai) "
        + _DATE,
        lambda m: _cur_amt(m, 3),
    ),
    _p(
        s.SALARY_NEXT_REDUCED,
        r"(?:next salary is reduced to) " + _CUR + r" " + _AMT,
        _cur_amt,
    ),
    _p(
        s.SALARY_TEMPORARY,
        r"(?:temporary monthly pay is|gaji bulanan sementara anda adalah) "
        + _CUR
        + r" "
        + _AMT,
        _cur_amt,
    ),
    _p(
        s.SALARY_PLUS_ARREARS,
        r"(?:regular salary for the next payroll is|gaji rutin anda untuk penggajian berikutnya adalah) "
        + _CUR
        + r" "
        + _AMT
        + r".*?(?:arrears adjustment of|tunggakan satu kali sebesar) "
        + _CUR
        + r" "
        + _AMT,
        lambda m: {
            "currency": m.group(1),
            "amount": _amount(m.group(2)),
            "secondary_amount": _amount(m.group(4)),
        },
    ),
    _p(
        s.SALARY_RESUMES,
        r"regular salary of " + _CUR + r" " + _AMT + r" resumes on " + _DATE,
        lambda m: _cur_amt(m, 3),
    ),
    _p(
        s.SALARY_CONFIRMED_FX,
        r"(?:your salary of|gaji sebesar) "
        + _CUR
        + r" "
        + _AMT
        + r" (?:is confirmed for|dikonfirmasi untuk) "
        + _DATE,
        lambda m: _cur_amt(m, 3),
    ),
    _p(
        s.FIRST_SALARY,
        r"(?:first salary will be|first salary from the new employer is|gaji pertama anda sebesar|"
        r"gaji pertama dari perusahaan baru adalah|first salary of) "
        + _CUR
        + r" "
        + _AMT
        + r".*?"
        + _DATE,
        lambda m: _cur_amt(m, 3),
    ),
    _p(
        s.INVOICE_APPROVED,
        r"(?:approved an invoice payment of|menyetujui pembayaran faktur sebesar) "
        + _CUR
        + r" "
        + _AMT
        + r".*?(?:settlement is expected on|penyelesaian diperkirakan pada) "
        + _DATE,
        lambda m: _cur_amt(m, 3),
    ),
    _p(
        s.BASE_SALARY_CONFIRMED,
        r"(?:confirmed base salary is|gaji pokok yang dikonfirmasi adalah) "
        + _CUR
        + r" "
        + _AMT,
        _cur_amt,
    ),
    _p(
        s.HOUSEHOLD_INCOME_REMAINING,
        r"(?:remaining confirmed monthly salary is|sisa gaji bulanan yang dikonfirmasi adalah) "
        + _CUR
        + r" "
        + _AMT,
        _cur_amt,
    ),
    _p(
        s.INCOME_ENDED,
        r"(seasonal contract has ended|kontrak musiman saat ini telah berakhir|employment has ended|"
        r"hubungan kerja anda telah berakhir)",
    ),
    # --- expenses -------------------------------------------------------------
    _p(
        s.RENT_INCREASE,
        r"(?:increases monthly rent by|menaikkan biaya sewa bulanan sebesar) (\d+(?:\.\d+)?)%",
        lambda m: {"percent": Decimal(m.group(1))},
    ),
    _p(s.FAILED_DEBIT_RETRY, r"previous debit attempt failed"),
    _p(
        s.INTERNAL_TRANSFER,
        r"(transfer between your two accounts|transfer antara dua rekening)",
    ),
    _p(s.SEPARATE_CARD_MINIMUMS, r"minimum payments due on two separate card accounts"),
    # --- confirmations that nothing should be counted -------------------------
    _p(
        s.BONUS_PENDING,
        r"(quarterly bonus is still subject|bonus kuartalan anda masih menunggu)",
    ),
    _p(
        s.GIG_PAYOUT_PENDING,
        r"(payout is still pending|pembayaran berikutnya dari \w+ masih tertunda)",
    ),
    _p(
        s.PRIZE_PENDING,
        r"(prize claim has been verified|klaim hadiah anda sudah diverifikasi)",
    ),
    _p(s.PRIZE_SETTLED, r"prize proceeds have reached your account"),
    _p(
        s.INVESTMENT_SALE_SETTLED,
        r"(proceeds from your investment sale|hasil penjualan investasi anda)",
    ),
    _p(
        s.PORTFOLIO_VALUE_CHANGE,
        r"(displayed market value|displayed value of the investment|nilai investasi yang ditampilkan)",
    ),
    _p(s.FX_REFUND_PROCESSING, r"foreign-currency refund is still processing"),
    _p(
        s.FX_CHARGE_PENDING,
        r"(charged in a foreign currency|tagihan dikenakan dalam mata uang asing)",
    ),
    _p(
        s.REFUND_PENDING,
        r"(refund has been initiated|pengembalian dana sudah diproses)",
    ),
    _p(s.DISPUTE_OPEN, r"(still being investigated|masih dalam penyelidikan)"),
    _p(
        s.REIMBURSEMENT_CLOSED,
        r"(reimbursement for your earlier work expense|penggantian atas biaya kerja)",
    ),
    _p(s.RECEIPT_CONFIRMATION, r"(receipt (has|contains) the final|paid in \w+ on)"),
]


def parse(message: Message) -> s.Amendment:
    text = " ".join(message.message_text.split())
    for intent, pattern, extractor in PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        fields = extractor(match) if callable(extractor) else {}
        return s.Amendment(
            message_id=message.message_id,
            user_id=message.user_id,
            intent=intent,
            related_event_id=message.related_event_id,
            source="rules",
            **fields,
        )
    return s.Amendment(
        message_id=message.message_id,
        user_id=message.user_id,
        intent=s.NO_EFFECT,
        related_event_id=message.related_event_id,
        source="rules",
    )
