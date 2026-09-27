"""Per-request working memory shared by the agent's nodes and tools.

Holds everything the agent has established so far - the evidence it recorded,
the forecast that evidence implies, every plan it tested, and the submission
the verifier accepted. The deterministic engine is rebuilt from this state on
demand, so each tool call sees exactly the facts the agent has committed to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from ..engine import Engine
from ..evidence import schema as s
from ..forecast import Forecaster
from ..ledger import Ledger
from ..models import Request
from ..plans import PlanSearch, search
from ..recurrence import ForecastRule
from ..writers import Decision


@dataclass
class Workspace:
    engine: Engine
    request: Request
    rule: ForecastRule

    amendments: list[s.Amendment] = field(default_factory=list)
    image_amounts: dict[str, Decimal] = field(default_factory=dict)
    image_readings: dict[str, dict] = field(default_factory=dict)

    lean_decisions: list[dict] = field(
        default_factory=list
    )  # one entry per model attempt
    rejected_submissions: list[str] = field(default_factory=list)
    decision: Decision | None = None
    decided_by: str = ""  # "agent" | "fallback"

    _forecaster: Forecaster | None = None
    _search: PlanSearch | None = None

    # -- accessors -------------------------------------------------------

    @property
    def data(self):
        return self.engine.data

    @property
    def profile(self):
        return self.data.profiles[self.request.user_id]

    @property
    def options(self):
        return self.data.options(self.request.request_id)

    @property
    def messages(self):
        return self.data.messages(self.request.user_id)

    def blank_amount_events(self):
        return [
            (event, self.data.images_by_event.get(event.event_id))
            for event in self.data.events(self.request.user_id)
            if event.amount is None
        ]

    # -- evidence --------------------------------------------------------

    def commit_evidence(
        self, amendments: list[s.Amendment], image_amounts: dict[str, Decimal]
    ) -> None:
        self.amendments = amendments
        self.image_amounts = image_amounts
        self._forecaster = None
        self._search = None

    # -- engine views ----------------------------------------------------

    def forecaster(self) -> Forecaster:
        if self._forecaster is None:
            ledger = Ledger(
                profile=self.profile,
                events=self.data.events(self.request.user_id),
                fx=self.engine.fx,
                as_of=self.request.request_date,
                image_amounts=self.image_amounts,
            )
            self._forecaster = Forecaster(ledger, self.rule, amendments=self.amendments)
        return self._forecaster

    def reference_search(self) -> PlanSearch:
        """Exhaustive deterministic plan search - used only to verify the agent."""
        if self._search is None:
            self._search = search(
                self.request, self.profile, self.options, self.forecaster()
            )
        return self._search

    @property
    def amount_safe_to_pay(self) -> Decimal:
        return self.reference_search().amount_safe_to_pay

    @property
    def earliest(self) -> date | None:
        return self.reference_search().earliest
