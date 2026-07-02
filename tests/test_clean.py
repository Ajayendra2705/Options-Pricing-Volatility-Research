"""
Day 6 tests: quote-cleaning pipeline (src/surface/clean.py).

Synthetic fixtures per filter (missing/zero-bid/crossed/wide/stale), union
semantics, mid/T columns, column normalization, and an end-to-end run over
the real raw parquet checking report consistency.
"""

import numpy as np
import pandas as pd
import pytest

from src.surface.clean import RAW_DIR, clean_chain, flag_quality, normalize_columns


def make_chain(**overrides):
    """One good baseline row; overrides create the defect under test."""
    base = dict(
        date="2023-06-02", expiration="2023-07-21", strike=100.0,
        call_put="Call", bid=5.0, ask=5.5, vol=10,
    )
    base.update(overrides)
    return pd.DataFrame([base])


GOOD = make_chain()


def test_good_row_survives_with_mid_and_T():
    clean, report = clean_chain(GOOD)
    assert len(clean) == 1
    assert report["total_dropped"] == 0
    assert clean.loc[0, "mid"] == pytest.approx(5.25)
    assert clean.loc[0, "T"] == pytest.approx(49 / 365.0)
    assert clean.loc[0, "option_type"] == "C"


@pytest.mark.parametrize(
    "defect,filter_name",
    [
        (dict(bid=np.nan), "missing"),
        (dict(ask=np.nan), "missing"),
        (dict(bid=0.0), "zero_bid"),
        (dict(bid=-1.0), "zero_bid"),
        (dict(bid=5.5, ask=5.5), "crossed"),
        (dict(bid=6.0, ask=5.5), "crossed"),
        (dict(bid=1.0, ask=2.0), "wide"),  # spread 1.0 / mid 1.5 = 67% > 50%
    ],
)
def test_single_filter_drops(defect, filter_name):
    clean, report = clean_chain(make_chain(**defect))
    assert len(clean) == 0
    assert report["filters"][filter_name] == 1
    assert report["total_dropped"] == 1
    assert report["drop_rate"] == 1.0


def test_stale_quote_flagged_second_day_only():
    rows = pd.concat(
        [
            make_chain(date="2023-06-02"),
            make_chain(date="2023-06-05"),                # identical bid/ask -> stale
            make_chain(date="2023-06-06", bid=5.1),       # bid moved -> fresh
        ],
        ignore_index=True,
    )
    flags = flag_quality(normalize_columns(rows))
    assert flags["stale"].tolist() == [False, True, False]
    clean, report = clean_chain(rows)
    assert report["filters"]["stale"] == 1
    assert len(clean) == 2


def test_stale_requires_same_contract():
    rows = pd.concat(
        [
            make_chain(date="2023-06-02", strike=100.0),
            make_chain(date="2023-06-05", strike=105.0),  # different contract, same quote
        ],
        ignore_index=True,
    )
    flags = flag_quality(normalize_columns(rows))
    assert not flags["stale"].any()


def test_expired_nonpos_T_dropped():
    clean, report = clean_chain(make_chain(expiration="2023-06-02"))  # T = 0
    assert len(clean) == 0
    assert report["expired_nonpos_T"] == 1


def test_union_semantics_no_double_count():
    # one row trips zero_bid AND wide: dropped once, counted per-filter
    clean, report = clean_chain(make_chain(bid=0.0, ask=3.0))
    assert report["filters"]["zero_bid"] == 1
    assert report["total_dropped"] == 1


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="missing required"):
        clean_chain(GOOD.drop(columns=["strike"]))


def test_put_normalization():
    clean, _ = clean_chain(make_chain(call_put="Put"))
    assert clean.loc[0, "option_type"] == "P"


def test_real_raw_chain_end_to_end():
    files = [f for f in RAW_DIR.glob("*.parquet")]
    if not files:
        pytest.skip("no raw data present")
    raw = pd.read_parquet(files[0])
    clean, report = clean_chain(raw)
    assert report["total_rows"] == len(raw)
    assert report["total_clean"] == len(clean)
    assert report["total_clean"] > 0
    assert report["drop_rate"] < 0.40                    # Day 1 gate still holds
    # invariants on the cleaned chain
    assert (clean["bid"] > 0).all()
    assert (clean["bid"] < clean["ask"]).all()
    assert (clean["T"] > 0).all()
    assert clean["mid"].between(clean["bid"], clean["ask"]).all()
    assert set(clean["option_type"].unique()) <= {"C", "P"}
