"""
Phase-2 (Day 32) — output-dir seams on the surface runners.
==========================================================
Phase 2 runs v1's surface code on SPY, so the runners gained `processed_dir` /
`plots_dir` / explicit results-json params. The whole point of the seam is that
v1 cannot move: these tests pin that

  1. every seam defaults to v1's module constant (a runner called with no args
     still reads and writes exactly where it did before), and
  2. a redirected run writes ONLY under the directories it was handed — no
     stray byte lands in data/processed/ or results/ (this repo has shipped
     that bug twice: Day-29 metrics.json, Day-30 data_quality.json).

(2) is asserted by sha256-snapshotting the v1 artifacts around a redirected
run, which is the same technique CI uses on pytest itself.
"""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import har, realized_vol
from src.backtest import signal as signal_mod
from src.surface import assemble, clean, forwards, iv_surface, no_arb, svi
from src.surface.clean import clean_chain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1_PROCESSED = PROJECT_ROOT / "data" / "processed"
V1_RESULTS = PROJECT_ROOT / "results"
V1_RAW = PROJECT_ROOT / "data" / "raw"

# runner -> the seam params it must expose
SEAMS = {
    forwards.run_forwards: ("processed_dir", "plots_dir", "make_plots"),
    iv_surface.run_iv_surface: ("processed_dir", "plots_dir", "make_plots"),
    svi.run_svi_all: ("processed_dir", "plots_dir", "make_plots"),
    no_arb.run_constrained_refit: ("processed_dir", "log_path"),
    no_arb.run_arb_check: ("processed_dir", "report_path"),
    assemble.run_assembly: ("processed_dir", "plots_dir", "qc_path", "make_plots"),
    clean.run_cleaning: ("raw_dir", "results_dir", "dq_path", "out_path"),
    # Day 33: the backtest-front runners got the same treatment
    realized_vol.run_realized_vol: ("processed_dir", "plots_dir", "make_plots"),
    har.run_har: ("processed_dir", "plots_dir", "stats_path", "make_plots"),
    signal_mod.run_signal: ("processed_dir", "plots_dir", "summary_path", "make_plots"),
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _snapshot(dirs) -> dict[Path, str]:
    return {f: _sha(f) for d in dirs if d.exists()
            for f in sorted(d.rglob("*")) if f.is_file()}


# ── (1) the seams exist and default to v1 ────────────────────────────────────

@pytest.mark.parametrize("fn,params", SEAMS.items(), ids=lambda x: getattr(x, "__name__", ""))
def test_seam_params_exist_and_default_to_none(fn, params):
    sig = inspect.signature(fn)
    for p in params:
        assert p in sig.parameters, f"{fn.__name__} lost seam {p!r}"
        default = sig.parameters[p].default
        if p == "make_plots":
            # defaults True so v1 keeps emitting its tracked figures
            expected = True
        elif p == "raw_dir":
            expected = clean.RAW_DIR          # bound eagerly since Day 6
        else:
            expected = None                   # resolved to the v1 constant inside
        assert default == expected, \
            f"{fn.__name__}.{p} default changed to {default!r}"


def test_module_constants_still_point_at_v1():
    # a seam that silently redirected the DEFAULT would move v1's artifacts
    for mod in (clean, forwards, iv_surface, svi, no_arb, assemble,
                realized_vol, har, signal_mod):
        assert mod.PROCESSED_DIR == V1_PROCESSED, mod.__name__
    assert clean.RAW_DIR == V1_RAW
    assert clean.RESULTS_DIR == V1_RESULTS
    assert realized_vol.RAW_DIR == V1_RAW


# ── (2) a redirected run touches nothing of v1's ─────────────────────────────

@pytest.fixture(scope="module")
def real_chain() -> pd.DataFrame:
    p = V1_PROCESSED / "chain_clean.parquet"
    if not p.exists():
        pytest.skip("cleaned chain not built (run main.py --stage clean)")
    return pd.read_parquet(p)


def test_redirected_run_writes_only_under_given_dirs(real_chain, tmp_path):
    """forwards -> ivs -> joint fit, entirely inside tmp_path."""
    before = _snapshot([V1_PROCESSED, V1_RESULTS])

    proc, plots = tmp_path / "processed", tmp_path / "plots"
    proc.mkdir()
    real_chain.to_parquet(proc / "chain_clean.parquet", index=False)

    forwards.run_forwards(processed_dir=proc, plots_dir=plots, make_plots=False)
    iv_surface.run_iv_surface(processed_dir=proc, plots_dir=plots, make_plots=False)
    no_arb.run_arb_check(processed_dir=proc, report_path=tmp_path / "arb.json")

    after = _snapshot([V1_PROCESSED, V1_RESULTS])
    assert after == before, (
        "redirected surface run mutated v1 artifacts: "
        f"{sorted(str(k) for k in set(before) ^ set(after)) or 'contents changed'}"
    )
    # and it did produce its outputs, in the handed-over dir
    assert (proc / "forwards.parquet").exists()
    assert (proc / "iv_surface.parquet").exists()
    assert (proc / "svi_params_joint.parquet").exists()
    assert (tmp_path / "arb.json").exists()
    assert not plots.exists(), "make_plots=False still wrote figures"


def test_calendar_floor_does_not_bind_outside_quoted_strikes():
    """Day 32: the floor is information only where the flooring slice is quoted.

    Built directly: a short slice quoted on |k| <= 0.1, whose SVI extrapolates
    to a large total variance at k = -0.5. The floor must be active inside the
    quoted range and inert outside it.
    """
    from src.surface.no_arb import _floor_mask, floor_on_grid
    from src.surface.svi import SVIParams, svi_total_variance

    kg = np.linspace(-1.5, 1.5, 1001)
    short = SVIParams(a=0.0005, b=0.05, rho=-0.85, m=0.0, sigma=0.1)
    floor = floor_on_grid(kg, short, k_lo=-0.1, k_hi=0.1)

    inside, outside = np.abs(kg) <= 0.1, np.abs(kg) > 0.1
    assert np.isfinite(floor[inside]).all(), "floor must bind where quoted"
    assert np.isneginf(floor[outside]).all(), "floor must not bind past quotes"
    np.testing.assert_allclose(floor[inside], svi_total_variance(kg[inside], short))

    # and the mask the fitter uses agrees
    active = _floor_mask(floor)
    assert active is not None and active.sum() == inside.sum()
    assert _floor_mask(None) is None
    assert _floor_mask(np.full_like(kg, -np.inf)) is None   # nothing quoted -> no floor


def test_a_floor_that_never_binds_does_not_change_the_fit():
    """Day 32, the real bug: `feasible` is not `good`.

    SLSQP on this objective has bad local minima, and handing it a constraint
    that is satisfied at the optimum could still push it into one — the fit was
    then accepted because it was feasible. Caught on SPY 2023-10-02/10-27: 9.68
    volpts under a floor that never binds, 0.34 with the floor absent. A
    non-binding constraint must cost nothing.
    """
    from src.surface.no_arb import K_SPAN, N_GRID, fit_svi_constrained
    from src.surface.svi import SVIParams, svi_total_variance

    p = V1_PROCESSED / "iv_surface.parquet"
    if not p.exists():
        pytest.skip("iv surface not built")
    surf = pd.read_parquet(p)
    ok = surf[surf["status"] == "ok"]
    date, expiry = ok.groupby(["date", "expiry"]).size().idxmax()
    sl = svi.otm_side(ok[(ok["date"] == date) & (ok["expiry"] == expiry)])
    k, iv = sl["log_moneyness"].to_numpy(), sl["iv"].to_numpy()
    T = float(sl["T"].iloc[0])

    _, free = fit_svi_constrained(k, iv, T, K_SPAN, N_GRID, w_floor=None)

    # a floor far below every market total variance: feasible everywhere, so it
    # carries no information and must not move the answer
    kg = np.linspace(-K_SPAN, K_SPAN, N_GRID)
    inert = svi_total_variance(kg, SVIParams(a=1e-9, b=1e-9, rho=0.0, m=0.0, sigma=0.1))
    _, floored = fit_svi_constrained(k, iv, T, K_SPAN, N_GRID, w_floor=inert)

    assert floored["floor_ok"]
    assert floored["rmse_iv"] == pytest.approx(free["rmse_iv"], abs=1e-4), (
        f"inert floor moved the fit: {free['rmse_iv']*100:.2f} -> "
        f"{floored['rmse_iv']*100:.2f} volpts")


def _chain_row(date, expiry="2024-01-19", strike=100.0, cp="C", bid=1.0, ask=1.1):
    return {"date": date, "expiry": expiry, "strike": strike,
            "option_type": cp, "bid": bid, "ask": ask}


def test_session_filter_drops_dates_with_no_underlying_bar():
    """Day 32: the DB quotes options on market holidays, carrying the previous
    session's chain forward. No bar = no hedge = not an observation date."""
    rows = []
    # quotes move day to day, so nothing here is dropped as stale: the session
    # filter is the only thing that can remove the holiday
    for j, d in enumerate(("2024-01-11", "2024-01-12", "2024-01-15")):  # 15th = MLK
        for i, K in enumerate((95.0, 100.0, 105.0)):
            rows.append(_chain_row(d, strike=K, bid=1.0 + i + 0.1 * j,
                                   ask=1.2 + i + 0.1 * j))
    chain = pd.DataFrame(rows)
    sessions = pd.Series(pd.to_datetime(["2024-01-11", "2024-01-12"]))  # no MLK

    clean, report = clean_chain(chain, sessions=sessions)

    assert set(clean["date"].dt.strftime("%Y-%m-%d")) == {"2024-01-11", "2024-01-12"}
    assert report["non_session_rows"] == 3
    assert report["dates_raw"] == 3 and report["dates_clean"] == 2
    assert report["total_rows"] == 9                    # denominator stays honest
    assert report["drop_rate"] == round(3 / 9, 4)


def test_session_filter_is_inert_when_not_asked_for():
    # v1 passes no sessions: its 5 AAPL dates are all real sessions
    rows = [_chain_row("2024-01-15", strike=K, bid=1.0 + i, ask=1.2 + i)
            for i, K in enumerate((95.0, 100.0, 105.0))]
    clean, report = clean_chain(pd.DataFrame(rows))
    assert report["non_session_rows"] == 0
    assert len(clean) == 3


def test_v1_arb_claims_are_scoped_to_quoted_strikes():
    """v1 checked +-1.0 with AAPL quoted to ~+-0.25: it claimed arb-freedom
    over 4x the range it had data for. The claim now states its domain."""
    qc = json.loads((V1_RESULTS / "surface_qc.json").read_text())
    assert "quoted" in qc["arb_checked_on"]
    for d in qc["dates"]:
        lo, hi = d["k_checked"]
        assert lo < 0 < hi                        # spans ATM
        assert hi - lo < 2.0                      # not the old fixed +-1.0 span


def test_redirected_cleaning_writes_dq_where_told(tmp_path):
    if not any(V1_RAW.rglob(clean.CHAIN_GLOB)):
        pytest.skip("raw chain not present")
    before = _snapshot([V1_PROCESSED, V1_RESULTS])

    dq = tmp_path / "data_quality_custom.json"
    out = clean.run_cleaning(raw_dir=V1_RAW, out_path=tmp_path / "chain_clean.parquet",
                             results_dir=tmp_path, dq_path=dq)

    assert out.parent == tmp_path
    assert "cleaning" in json.loads(dq.read_text())
    assert not (tmp_path / "data_quality.json").exists(), "dq_path ignored"
    assert _snapshot([V1_PROCESSED, V1_RESULTS]) == before
