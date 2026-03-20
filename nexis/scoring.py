"""Scoring and weight computation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .protocol import FAILURE_LOOKBACK_INTERVALS, SCORING_EXPONENT


@dataclass
class MinerIntervalScore:
    miner_hotkey: str
    interval_id: int
    accepted: bool
    passed_sample_count: int

    @property
    def score(self) -> float:
        if not self.accepted:
            return 0.0
        return float(pow(self.passed_sample_count, SCORING_EXPONENT))


class WeightComputer:
    """Normalizes miner scores with failure lookback gating."""

    def __init__(self, lookback: int = FAILURE_LOOKBACK_INTERVALS):
        self._lookback = lookback
        self._history: deque[dict[str, bool]] = deque(maxlen=lookback)

    def update_failure_history(self, decisions: dict[str, bool]) -> None:
        """decisions: hotkey -> accepted."""
        self._history.append(decisions)

    def has_recent_failure(self, hotkey: str) -> bool:
        for window in self._history:
            accepted = window.get(hotkey)
            if accepted is False:
                return True
        return False

    def compute_weights(self, interval_scores: list[MinerIntervalScore]) -> dict[str, float]:
        raw: dict[str, float] = defaultdict(float)
        for row in interval_scores:
            if self.has_recent_failure(row.miner_hotkey):
                raw[row.miner_hotkey] = 0.0
                continue
            raw[row.miner_hotkey] += row.score

        return self.normalize_weights(raw)

    def compute_weights_from_totals(self, score_totals: dict[str, float]) -> dict[str, float]:
        """Normalize already-accumulated score totals with failure gating."""
        raw: dict[str, float] = defaultdict(float)
        for hotkey, total_score in score_totals.items():
            if self.has_recent_failure(hotkey):
                raw[hotkey] = 0.0
                continue
            raw[hotkey] += float(total_score)
        return self.normalize_weights(raw)

    def normalize_weights(self, raw_scores: dict[str, float]) -> dict[str, float]:
        total = sum(raw_scores.values())
        if total <= 0:
            return {hotkey: 0.0 for hotkey in raw_scores}
        return {hotkey: score / total for hotkey, score in raw_scores.items()}

    @staticmethod
    def score_from_sample_count(sample_count: int) -> float:
        if sample_count <= 0:
            return 0.0
        return float(pow(sample_count, SCORING_EXPONENT))


