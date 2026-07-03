"""
Day 18 tests: signal construction (pre-registered primary).

Synthetic SVI params (b=0 -> w(0)=a, so ATM IV is exact by construction) +
synthetic OHLC drive the whole path: ATM IV extraction, horizon matching,
signal arithmetic vs a manual recompute, rank/side assignment, trailing-only
z-score, and exclusion rules (fit_ok=False, forecast unavailable).
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.har import expanding_forecast, har_dataset
from src.backtest.signal import (
    MIN_Z_OBS,
    atm_iv,
    build_signal,
    horizon,
    tenor_bucket,
)
from tests.test_realized_vol import gbm_ohlc


@pytest.fixture(scope="module")
def ohlc():
    return gbm_ohlc()


def _params(dates, ivs_by_T):
    """SVI param rows with b=0 => atm_iv == sqrt(a/T) exactly."""
    rows = []
    for d, per_date in zip(dates, ivs_by_T):
        for T, iv in per_date:
            rows.append({
                "date": d, "expiry": d + pd.Timedelta(days=int(T * 365)),
                "T": T, "a": iv ** 2 * T, "b": 0.0, "rho": 0.0, "m": 0.0,
                "sigma": 0.1, "fit_ok": True,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def quote_dates(ohlc):
    # deep enough into the sample that every horizon's expanding HAR exists
    return [ohlc["date"].iloc[i] for i in (300, 320, 340, 360, 380)]


@pytest.fixture(scope="module")
def params(quote_dates):
    # 3 tenors per date, IV ladder varies by date
    return _params(
        quote_dates,
        [[(0.04, 0.30), (0.08, 0.28), (0.13, 0.26)],
         [(0.04, 0.20), (0.08, 0.31), (0.13, 0.27)],
         [(0.04, 0.25), (0.08, 0.25), (0.13, 0.25)],
         [(0.04, 0.35), (0.08, 0.22), (0.13, 0.28)],
         [(0.04, 0.24), (0.08, 0.26), (0.13, 0.30)]],
    )


def test_atm_iv_exact_for_flat_svi():
    row = pd.Series({"a": 0.30 ** 2 * 0.08, "b": 0.0, "rho": 0.0, "m": 0.0,
                     "sigma": 0.1, "T": 0.08})
    assert atm_iv(row) == pytest.approx(0.30, rel=1e-12)


def test_bucket_and_horizon():
    assert tenor_bucket(0.04) == "short" and tenor_bucket(0.08) == "mid"
    assert tenor_bucket(0.13) == "long"
    assert horizon(0.04) == 10 and horizon(0.08) == 20 and horizon(0.13) == 33
    assert horizon(0.005) == 5          # clamp low
    assert horizon(0.50) == 63          # clamp high


def test_signal_raw_matches_manual(params, ohlc):
    tab = build_signal(params, ohlc)
    r = tab.iloc[0]
    fcst = expanding_forecast(har_dataset(ohlc, r["h"]), r["h"])
    assert np.isfinite(r["signal_raw"])
    assert r["signal_raw"] == pytest.approx(r["atm_iv"] - fcst[r["date"]], abs=1e-15)


def test_rank_and_sides(params, ohlc):
    tab = build_signal(params, ohlc)
    for _, day in tab.groupby("date"):
        assert sorted(day["rank"]) == [1.0, 2.0, 3.0]
        assert day.loc[day["rank"] == 1.0, "side"].item() == "short_vol"
        assert day.loc[day["rank"] == 3.0, "side"].item() == "long_vol"
        assert day.loc[day["rank"] == 2.0, "side"].item() == "flat"
        # richest slice really has the highest signal
        d = day.set_index("side")["signal_raw"]
        assert d["short_vol"] == day["signal_raw"].max()
        assert d["long_vol"] == day["signal_raw"].min()


def test_z_trailing_only(params, ohlc, quote_dates):
    tab = build_signal(params, ohlc)
    # first MIN_Z_OBS dates per bucket lack enough prior history
    for _, b in tab.groupby("bucket"):
        b = b.sort_values("date")
        assert b["signal_z"].iloc[:MIN_Z_OBS].isna().all()
        assert b["signal_z"].iloc[MIN_Z_OBS:].notna().all()
    # no lookahead: dropping the last two quote dates changes nothing before
    trunc = build_signal(params[params["date"] <= quote_dates[2]], ohlc)
    keep = tab[tab["date"] <= quote_dates[2]].reset_index(drop=True)
    pd.testing.assert_frame_equal(keep, trunc)


def test_unfitted_slices_dropped(params, ohlc):
    p = params.copy()
    p.loc[p.index[0], "fit_ok"] = False
    tab = build_signal(p, ohlc)
    assert len(tab) == len(params) - 1


def test_unavailable_forecast_is_flat(params, ohlc, quote_dates):
    # a quote date before the HAR warm-up: forecast NaN -> excluded from
    # ranking, side flat; the remaining slices still rank 1..2
    early = _params([ohlc["date"].iloc[30]], [[(0.04, 0.30)]])
    p = pd.concat([early, params], ignore_index=True)
    tab = build_signal(p, ohlc)
    row = tab[tab["date"] == ohlc["date"].iloc[30]].iloc[0]
    assert np.isnan(row["signal_raw"]) and row["side"] == "flat"
    assert np.isnan(row["rank"])
