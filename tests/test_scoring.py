from __future__ import annotations

from nexis.scoring import MinerIntervalScore, WeightComputer


def test_score_from_sample_count_matches_interval_score_semantics() -> None:
    computer = WeightComputer()
    assert computer.score_from_sample_count(0) == 0.0
    assert computer.score_from_sample_count(3) == MinerIntervalScore(
        miner_hotkey="hk",
        interval_id=1,
        accepted=True,
        passed_sample_count=3,
    ).score


def test_compute_weights_from_totals_normalizes() -> None:
    computer = WeightComputer()
    weights = computer.compute_weights_from_totals({"hk1": 9.0, "hk2": 3.0})
    assert weights["hk1"] == 0.75
    assert weights["hk2"] == 0.25


def test_compute_weights_from_totals_applies_failure_gating() -> None:
    computer = WeightComputer()
    computer.update_failure_history({"hk1": False, "hk2": True})
    weights = computer.compute_weights_from_totals({"hk1": 9.0, "hk2": 3.0})
    assert weights["hk1"] == 0.0
    assert weights["hk2"] == 1.0
