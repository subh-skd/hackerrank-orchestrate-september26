"""Wires dataset, evidence and forecast together for one request.

This is the single place where extracted evidence (image amounts, message
amendments) enters the deterministic pipeline, so the calibration harness and
the production run see exactly the same inputs.
"""

from __future__ import annotations

from decimal import Decimal

from .forecast import Forecaster
from .fx import FxTable
from .ledger import Ledger
from .models import Dataset, Request
from .recurrence import ForecastRule


class Engine:
    def __init__(
        self,
        data: Dataset,
        image_amounts: dict[str, Decimal] | None = None,
        amendments: dict[str, list] | None = None,
    ) -> None:
        self.data = data
        self.fx = FxTable(data.rates)
        self.image_amounts = image_amounts or {}
        # event_id -> audit record of the image reading (OCR lines, model reply).
        self.image_records: dict[str, dict] = {}
        # user_id -> list[Amendment]; applied by the forecaster (evidence layer).
        self.amendments = amendments or {}
        self._ledgers: dict[str, Ledger] = {}

    def ledger(self, request: Request) -> Ledger:
        cached = self._ledgers.get(request.request_id)
        if cached is None:
            cached = Ledger(
                profile=self.data.profiles[request.user_id],
                events=self.data.events(request.user_id),
                fx=self.fx,
                as_of=request.request_date,
                image_amounts=self.image_amounts,
            )
            self._ledgers[request.request_id] = cached
        return cached

    def forecaster(self, request: Request, rule: ForecastRule) -> Forecaster:
        return Forecaster(
            self.ledger(request),
            rule,
            amendments=self.amendments.get(request.user_id, []),
        )
