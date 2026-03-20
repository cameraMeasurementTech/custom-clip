from __future__ import annotations

from nexis.cli import (
    _initial_miner_interval_start,
    _latest_eligible_validation_interval_start,
    _weight_retry_backoff_sec,
)


def test_initial_miner_interval_start_backfills_previous_interval() -> None:
    assert _initial_miner_interval_start(7756977) == 7756900
    assert _initial_miner_interval_start(7757000) == 7756950
    assert _initial_miner_interval_start(10) == 0


def test_latest_eligible_validation_interval_start_respects_reserve() -> None:
    # Reserve=2 blocks means interval 7756900-7756950 is eligible at block 7756952.
    assert _latest_eligible_validation_interval_start(7756950) == 7756850
    assert _latest_eligible_validation_interval_start(7756951) == 7756850
    assert _latest_eligible_validation_interval_start(7756952) == 7756900


def test_weight_retry_backoff_is_exponential_and_capped() -> None:
    assert _weight_retry_backoff_sec(1) == 10
    assert _weight_retry_backoff_sec(2) == 20
    assert _weight_retry_backoff_sec(3) == 40
    assert _weight_retry_backoff_sec(10) == 300


