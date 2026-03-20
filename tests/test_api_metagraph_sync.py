from __future__ import annotations

from types import SimpleNamespace

from nexis.api.metagraph_sync import extract_hotkeys_with_min_stake


def test_extract_hotkeys_with_min_stake_filters_values() -> None:
    metagraph = SimpleNamespace(
        hotkeys=["hk1", "hk2", "hk3", ""],
        S=[4999.0, 5000.0, 7500.5, 9999.0],
    )

    result = extract_hotkeys_with_min_stake(metagraph=metagraph, min_stake=5000.0)

    assert result == {
        "hk2": 5000.0,
        "hk3": 7500.5,
    }

