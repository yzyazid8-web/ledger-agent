import pytest

from ledger_agent.ingestion.universe import load_universe, select_companies


def test_universe_covers_financebench_companies() -> None:
    universe = load_universe()

    assert len(universe) == 32
    assert len({c.ticker for c in universe}) == 32
    assert len({c.cik for c in universe}) == 32


def test_cik_is_zero_padded() -> None:
    (mmm,) = select_companies(["mmm"])

    assert mmm.cik10 == "0000066740"


def test_unknown_ticker_is_rejected() -> None:
    with pytest.raises(ValueError, match="NOPE"):
        select_companies(["MMM", "NOPE"])
