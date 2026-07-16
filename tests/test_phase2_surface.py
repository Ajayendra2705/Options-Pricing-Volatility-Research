"""
Phase-2 (Day 32) — SPY surface gate tests.
==========================================
Day 31 proved the SPY *data* is real; this is the gate on the *surface* built
from it by v1's own code (scripts/run_phase2_surface.py). The assertions are
the ones v1 shipped on — arb-free after the joint fit, and the fitted surface
tracking the market inside a vol point — held to the same bar on 31x the data,
so a Phase-2 surface that is quietly worse than v1's cannot pass.

Skips wholesale if the Phase-2 surface has not been built (mirrors the v1
real-data tests), so a checkout without it stays green.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
P2_PROCESSED = PROJECT_ROOT / "data" / "phase2" / "processed"
P2_RESULTS = PROJECT_ROOT / "results" / "phase2"

QC_JSON = P2_RESULTS / "surface_qc_spy.json"
ARB_JSON = P2_RESULTS / "arb_violations_spy.json"
DQ_JSON = P2_RESULTS / "data_quality_spy.json"
JOINT = P2_PROCESSED / "svi_params_joint.parquet"

# 155 raw quote dates minus the 8 US market holidays the DB quotes anyway
# (no bar for the underlying = not a session = not an observation date, Day 32)
N_SESSIONS = 147

pytestmark = pytest.mark.skipif(
    not (QC_JSON.exists() and ARB_JSON.exists()),
    reason="Phase-2 SPY surface not built (run scripts/run_phase2_surface.py)",
)


@pytest.fixture(scope="module")
def qc() -> dict:
    return json.loads(QC_JSON.read_text())


@pytest.fixture(scope="module")
def arb() -> dict:
    return json.loads(ARB_JSON.read_text())


# ── coverage: the whole pre-registered window got a surface ──────────────────

def test_surface_covers_every_trading_session(qc):
    assert qc["n_dates"] == N_SESSIONS
    # 3 expiries/date is the DB's shape (Day-31)
    assert qc["n_slices_total"] >= 3 * N_SESSIONS - 2


def test_no_market_holiday_reached_the_surface(qc):
    # the 4 holidays the stale filter missed (Good Friday, Juneteenth, New
    # Year, MLK) used to build surfaces on days the market was shut
    holidays = {"2023-09-04", "2023-12-25", "2024-01-01", "2024-01-15",
                "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19"}
    assert not holidays & {d["date"] for d in qc["dates"]}


def test_joint_fit_converged_on_every_slice(arb):
    assert arb["n_slices_fitted"] >= 3 * N_SESSIONS - 2


# ── the v1 claims, held on SPY ───────────────────────────────────────────────

def test_no_butterfly_arbitrage(arb, qc):
    assert arb["butterfly"]["n_violations"] == 0
    assert arb["butterfly"]["min_g_across_slices"] >= 0.0
    assert qc["all_interp_butterfly_ok"] is True          # incl. between nodes
    assert qc["worst_interp_min_g"] >= 0.0


def test_no_calendar_arbitrage(arb, qc):
    assert arb["calendar"]["n_pairs_violated"] == 0
    assert arb["calendar"]["max_severity_w"] == 0.0
    assert arb["calendar"]["n_floor_failures_in_fit"] == 0
    assert qc["all_interp_calendar_ok"] is True


def test_surface_tracks_market_to_v1_standard(qc):
    # v1's Day-14 gate: >95% of market quotes within 1 vol point of the surface
    assert qc["frac_within_1volpt"] > 0.95
    assert qc["rmse_iv_median"] < 0.01                     # < 1 volpt median RMSE


def test_every_date_meets_the_gate(qc, inverted_dates):
    # the aggregate can hide a bad day across 147 of them — check per-date
    bad = [d for d in qc["dates"]
           if not (d["interp_butterfly_ok"] and d["interp_calendar_ok"])]
    assert not bad, [d["date"] for d in bad]
    # dates where the MARKET's own total variance falls with maturity are
    # exempt: an arb-free fit has to lift the long slice off those quotes, and
    # the residual is the honest price of the no-arb constraint (see
    # test_floor_only_costs_where_the_market_is_inverted)
    clean = [d for d in qc["dates"] if d["date"] not in inverted_dates]
    worst = min(d["frac_within_1volpt"] for d in clean)
    assert worst > 0.80, worst
    # inversions are a property of SPY's quotes, not of a handful of odd days —
    # but they must stay a minority, or the exemption above would be vacuous
    assert len(inverted_dates) < 0.2 * qc["n_dates"], len(inverted_dates)


INVERSION_TOL_VOLPTS = 0.5     # below this the "inversion" is fitting noise


@pytest.fixture(scope="module")
def inverted_dates() -> set[str]:
    """Dates whose QUOTES violate calendar monotonicity in total variance.

    Measured off the per-slice (butterfly-only, unfloored) fits, on the k range
    where both slices are quoted: w_short > w_long there means the market's own
    prices are calendar-inverted, so no arb-free surface can match them and the
    floor legitimately has to lift the long slice.

    Note the lift CASCADES: once a slice is raised off inverted quotes, every
    longer slice on that date must clear the raised one, so a later pair can
    carry cost without being inverted itself. Hence the exemption is per DATE.
    """
    from src.surface.svi import svi_total_variance

    per_slice = pd.read_parquet(P2_PROCESSED / "svi_params_constrained.parquet")
    per_slice = per_slice[per_slice["fit_ok"]]
    ivs = pd.read_parquet(P2_PROCESSED / "iv_surface.parquet")
    ivs = ivs[ivs["status"] == "ok"]

    out = set()
    for date, day in per_slice.groupby("date"):
        rows = list(day.sort_values("T").itertuples())
        for s, l in zip(rows, rows[1:]):
            gs = ivs[(ivs["date"] == date) & (ivs["expiry"] == s.expiry)]
            gl = ivs[(ivs["date"] == date) & (ivs["expiry"] == l.expiry)]
            lo = max(gs["log_moneyness"].min(), gl["log_moneyness"].min())
            hi = min(gs["log_moneyness"].max(), gl["log_moneyness"].max())
            if not (hi > lo):
                continue
            kg = np.linspace(lo, hi, 200)
            ws = svi_total_variance(kg, (s.a, s.b, s.rho, s.m, s.sigma))
            wl = svi_total_variance(kg, (l.a, l.b, l.rho, l.m, l.sigma))
            i = int(np.argmax(ws - wl))
            if (ws - wl)[i] <= 0:
                continue
            # size it in vol points on the long slice, not in raw variance
            volpts = (np.sqrt(max(ws[i], 0) / l.T)
                      - np.sqrt(max(wl[i], 0) / l.T)) * 100
            if volpts > INVERSION_TOL_VOLPTS:
                out.add(str(pd.Timestamp(date).date()))
    return out


def test_floor_only_costs_where_the_market_is_inverted(inverted_dates):
    """The Day-32 bugs, pinned.

    Two things made the joint fit worse than the per-slice fit for no good
    reason: (1) the calendar floor was the previous slice's SVI EXTRAPOLATED
    past its quoted strikes (2023-10-09 fitted at 12.74 volpts vs 0.18
    unfloored); (2) a floor that never binds could still push SLSQP into a bad
    local minimum that was accepted because it was feasible (2023-10-02: 9.68
    vs 0.34). Both are fixed, so the floor may only cost where it legitimately
    binds: on dates whose quotes are themselves calendar-inverted.
    """
    joint = pd.read_parquet(JOINT)
    per_slice = pd.read_parquet(P2_PROCESSED / "svi_params_constrained.parquet")
    m = (joint[joint["fit_ok"]]
         .merge(per_slice[per_slice["fit_ok"]][["date", "expiry", "rmse_iv"]],
                on=["date", "expiry"], suffixes=("_joint", "_perslice")))
    m["cost"] = (m["rmse_iv_joint"] - m["rmse_iv_perslice"]) * 100
    m["day"] = m["date"].map(lambda d: str(pd.Timestamp(d).date()))

    clean = m[~m["day"].isin(inverted_dates)]
    worst = clean.nlargest(1, "cost").iloc[0]
    assert clean["cost"].max() < 1.0, (
        f"floor costs {worst['cost']:.2f} volpts on {worst['day']} exp "
        f"{pd.Timestamp(worst['expiry']).date()}, where the market is NOT inverted")

    # the exempt dates are real inversions, not a blanket excuse: they are few
    # and the cost is bounded
    assert m["cost"].max() < 5.0, m.nlargest(1, "cost")[["day", "cost"]].to_dict()


def test_arb_claims_are_scoped_to_quoted_strikes(qc, arb):
    # the claim must state its domain, not imply the whole real line
    assert "quoted" in qc["arb_checked_on"]
    assert "quoted" in arb["calendar"]["checked_on"]
    lo, hi = qc["narrowest_k_checked"]
    assert lo < 0 < hi                      # every date's domain spans ATM


# ── isolation: Phase 2 wrote only Phase-2 paths ──────────────────────────────

def test_cleaning_block_appended_not_clobbering_day31_audit():
    dq = json.loads(DQ_JSON.read_text())
    assert dq["gate_decision"] == "PROCEED"                # Day-31 block survives
    assert dq["cleaning"]["total_clean"] > 10_000          # Day-32 block added
    assert "phase2/processed" in dq["cleaning"]["output"]


def test_v1_surface_untouched():
    v1 = json.loads((PROJECT_ROOT / "results" / "surface_qc.json").read_text())
    assert v1["n_dates"] == 5                              # v1's AAPL window
    assert v1["n_slices_total"] == 15


def test_joint_params_are_spy_window():
    fits = pd.read_parquet(JOINT)
    dates = pd.to_datetime(fits["date"])
    assert dates.min() >= pd.Timestamp("2023-07-01")
    assert dates.max() <= pd.Timestamp("2024-06-30")
