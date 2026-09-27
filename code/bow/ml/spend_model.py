"""Learned conservative forecast of essential variable spending.

The statement asks for essential variable spending to be forecast
"conservatively". Instead of hand-picking a statistic, this learns the
distribution of the *next* grocery / transport / dining occurrence from every
user's settled history and predicts an upper quantile of it.

Self-supervised training (no labels, no organizer data): for each series and
each occurrence i with enough prior history, the features describe the
previous occurrences and the target is occurrence i. Amounts are normalized by
the series' running median, so one model serves IDR and EUR users alike.

Model: scikit-learn HistGradientBoostingRegressor with quantile loss, one per
quantile. A time-ordered holdout (each series' most recent occurrence) is
scored with pinball loss against statistical baselines, and the result is
written to evaluation/ml_report.md.
"""

from __future__ import annotations

import pickle
import statistics
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np

CATEGORIES = ("groceries", "transport", "dining")
QUANTILES = (0.5, 0.75, 0.9)
WINDOW = 6  # prior occurrences used as features


@dataclass
class Series:
    user_id: str
    category: str
    points: list[tuple[date, float]]


def collect_series(data) -> list[Series]:
    out = []
    for user_id, events in data.events_by_user.items():
        for category in CATEGORIES:
            points = sorted(
                (e.cash_date, float(e.amount))
                for e in events
                if e.category == category
                and e.status == "settled"
                and e.direction == "debit"
                and e.amount is not None
                and e.amount > 0
            )
            if len(points) >= WINDOW + 1:
                out.append(Series(user_id, category, points))
    return out


def features(
    history: list[tuple[date, float]], category: str, next_day: date | None = None
) -> tuple[list[float], float]:
    """Feature vector from the last WINDOW occurrences, and the normalizer."""
    recent = history[-WINDOW:]
    amounts = [a for _d, a in recent]
    scale = statistics.median(a for _d, a in history) or 1.0
    ratios = [a / scale for a in amounts]
    gaps = [(b[0] - a[0]).days for a, b in zip(recent, recent[1:])] or [7]
    upcoming = next_day or (recent[-1][0])
    row = [
        *ratios,
        float(np.mean(ratios)),
        float(np.std(ratios)),
        float(max(ratios)),
        float(min(ratios)),
        float(np.median(gaps)),
        float(len(history)),
        float(upcoming.weekday()),
        float(upcoming.day),
        *[1.0 if category == c else 0.0 for c in CATEGORIES],
    ]
    return row, scale


def build_dataset(series: list[Series]):
    X_train, y_train, X_test, y_test, test_meta = [], [], [], [], []
    for s in series:
        for i in range(WINDOW, len(s.points)):
            history = s.points[:i]
            row, scale = features(history, s.category, s.points[i][0])
            target = s.points[i][1] / scale
            if i == len(s.points) - 1:
                X_test.append(row)
                y_test.append(target)
                test_meta.append((history, s.points[i][1], scale))
            else:
                X_train.append(row)
                y_train.append(target)
    return (
        np.array(X_train),
        np.array(y_train),
        np.array(X_test),
        np.array(y_test),
        test_meta,
    )


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


class SpendQuantileModel:
    def __init__(self) -> None:
        self.models: dict[float, object] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpendQuantileModel":
        from sklearn.ensemble import HistGradientBoostingRegressor

        for q in QUANTILES:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                random_state=7,
            )
            model.fit(X, y)
            self.models[q] = model
        return self

    def predict_ratio(self, X: np.ndarray, q: float) -> np.ndarray:
        return self.models[q].predict(X)

    def predict_amount(
        self, history: list[tuple[date, float]], category: str, next_day: date, q: float
    ) -> Decimal:
        row, scale = features(history, category, next_day)
        ratio = float(self.predict_ratio(np.array([row]), q)[0])
        return Decimal(str(round(max(ratio, 0.0) * scale, 2)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.models, handle)

    @classmethod
    def load(cls, path: Path) -> "SpendQuantileModel":
        model = cls()
        with path.open("rb") as handle:
            model.models = pickle.load(handle)
        return model


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "spend_quantile.pkl"
_LOADED: SpendQuantileModel | None = None


def get_model() -> SpendQuantileModel | None:
    global _LOADED
    if _LOADED is None and MODEL_PATH.exists():
        _LOADED = SpendQuantileModel.load(MODEL_PATH)
    return _LOADED


def train_and_report(data, report_path: Path) -> SpendQuantileModel:
    series = collect_series(data)
    X_train, y_train, X_test, y_test, meta = build_dataset(series)
    model = SpendQuantileModel().fit(X_train, y_train)
    model.save(MODEL_PATH)

    # Baselines on the same holdout, in the same normalized units.
    last = np.array([h[-1][1] / s for h, _y, s in meta])
    mean4 = np.array([np.mean([a for _d, a in h[-4:]]) / s for h, _y, s in meta])
    max4 = np.array([np.max([a for _d, a in h[-4:]]) / s for h, _y, s in meta])

    lines = [
        "# ML report: essential-spending quantile model",
        "",
        "Self-supervised `HistGradientBoostingRegressor` (quantile loss) that predicts the next grocery, "
        "transport or dining amount from the previous six occurrences, normalized by each series' median. "
        "It is trained on every user's settled history; no answer labels or organizer files are used.",
        "",
        f"- Series: {len(series):,} · training rows: {len(X_train):,} · holdout rows "
        f"(each series' most recent occurrence): {len(X_test):,}",
        "",
        "## Holdout pinball loss (lower is better)",
        "",
        "| Quantile | Model | Last value | Mean of last 4 | Max of last 4 | Holdout coverage (actual ≤ prediction) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for q in QUANTILES:
        pred = model.predict_ratio(X_test, q)
        coverage = float(np.mean(y_test <= pred))
        lines.append(
            f"| {q} | **{pinball(y_test, pred, q):.4f}** | {pinball(y_test, last, q):.4f} | "
            f"{pinball(y_test, mean4, q):.4f} | {pinball(y_test, max4, q):.4f} | {coverage:.1%} |"
        )
    lines += [
        "",
        "Coverage near the nominal quantile means the model is calibrated; a q=0.75 forecast should cover "
        'about 75% of next occurrences. That is the operational meaning of forecasting "conservatively".',
        "",
        "Whether the forecast uses this model (`variable_estimator = ml_q50 | ml_q75 | ml_q90`) or a "
        "statistical estimator is decided by the calibration search on the solved samples; see "
        "`calibration_report.md`.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return model
