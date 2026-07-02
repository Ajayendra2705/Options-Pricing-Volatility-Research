"""
Day 7 tests: parity-implied forwards (ATM-window median, external rate).

- Synthetic chain (exact Black-76 mids, known F/r): recover F to near
  machine precision; per-strike forwards identical (F_std ~ 0).
- Rate misspecification: r off by ±200bp moves F by < 1c near ATM
  (the reason for the ATM-window design).
- Quote noise: F within tolerance, diagnostics reflect noise.
- Carry from forward term structure: recovers known q without spot.
- Real cleaned chain: F stable across strikes per expiry (PLAN.md test),
  levels in the June-2023 AAPL band, carry sane.
- Degenerate slices return None.
"""

import numpy as np
import pandas as pd
import pytest

from src.greeks import black_scholes as bs
from src.surface.clean import PROCESSED_DIR
from src.surface.forwards import implied_carry, imply_forward_slice, imply_forwards


def synth_slice(F=185.0, r=0.05, T=0.4, sigma=0.25, strikes=None, noise=0.0, seed=0,
                expiry="2023-06-16"):
    """Exact European chain slice: call+put mids at each strike."""
    strikes = np.asarray(strikes if strikes is not None else np.arange(150.0, 220.0, 5.0))
    rng = np.random.default_rng(seed)
    rows = []
    for cp, name in ((1, "C"), (-1, "P")):
        mids = bs.price(F, strikes, T, sigma, r, cp)
        mids = mids + noise * rng.standard_normal(len(strikes))
        for K, m in zip(strikes, mids):
            rows.append({"date": "2023-06-02", "expiry": expiry, "strike": K,
                         "option_type": name, "mid": m, "T": T})
    return pd.DataFrame(rows)


def test_exact_recovery():
    F, r, T = 185.0, 0.05, 0.4
    res = imply_forward_slice(synth_slice(F=F, r=r, T=T), r=r)
    assert res is not None
    assert res["F"] == pytest.approx(F, rel=1e-10)
    assert res["F_std"] == pytest.approx(0.0, abs=1e-8)   # parity exact per strike
    assert res["r_implied"] == pytest.approx(r, abs=1e-6)  # diagnostic honest on European data


def test_rate_misspecification_barely_moves_F():
    F, r_true, T = 185.0, 0.05, 0.4
    sl = synth_slice(F=F, r=r_true, T=T)
    for r_wrong in (r_true - 0.02, r_true + 0.02):        # ±200bp
        res = imply_forward_slice(sl, r=r_wrong)
        assert res["F"] == pytest.approx(F, abs=0.02)     # < 2 cents on a $185 forward


def test_noisy_recovery_within_tolerance():
    F = 185.0
    res = imply_forward_slice(synth_slice(F=F, noise=0.02, seed=42), r=0.05)
    assert res is not None
    assert res["F"] == pytest.approx(F, abs=0.10)
    assert res["F_std"] > 0                                # noise shows up in diagnostics


def test_too_few_pairs_returns_none():
    assert imply_forward_slice(synth_slice(strikes=[180.0, 185.0])) is None


def test_missing_puts_returns_none():
    sl = synth_slice()
    assert imply_forward_slice(sl[sl["option_type"] == "C"]) is None


def test_imply_forwards_groups_slices():
    a = synth_slice(F=185.0, T=0.4)
    b = synth_slice(F=186.0, T=0.9, expiry="2023-12-15")
    fwd = imply_forwards(pd.concat([a, b], ignore_index=True), r=0.05)
    assert len(fwd) == 2
    np.testing.assert_allclose(fwd["F"], [185.0, 186.0], rtol=1e-9)


def test_carry_recovered_from_term_structure():
    # F(T) = S * exp((r - q) T): build 4 expiries with known q, no spot given
    S, r, q = 185.0, 0.05, 0.012
    slices = []
    for i, T in enumerate((0.1, 0.25, 0.5, 1.0)):
        F_T = S * np.exp((r - q) * T)
        slices.append(synth_slice(F=F_T, r=r, T=T, expiry=f"2023-1{i}-15"))
    fwd = imply_forwards(pd.concat(slices, ignore_index=True), r=r)
    carry = implied_carry(fwd)
    assert len(carry) == 1
    assert carry.loc[0, "q_implied"] == pytest.approx(q, abs=1e-6)


# --- Real data (PLAN.md Day 7 test: F stable across strikes per expiry) -----

@pytest.fixture(scope="module")
def real_forwards():
    path = PROCESSED_DIR / "chain_clean.parquet"
    if not path.exists():
        pytest.skip("cleaned chain not built (run main.py --stage clean)")
    return imply_forwards(pd.read_parquet(path))


def test_real_slices_produced(real_forwards):
    assert len(real_forwards) >= 10


def test_real_forward_stability_across_strikes(real_forwards):
    # ATM-window forwards agree to < 0.5% of F on every slice;
    # all-strike spread stays < 1.5% (American wings tolerated, quantified)
    atm_rel = real_forwards["F_std_atm"] / real_forwards["F"]
    all_rel = real_forwards["F_std"] / real_forwards["F"]
    assert atm_rel.max() < 0.005, real_forwards.loc[atm_rel.idxmax()].to_dict()
    assert all_rel.max() < 0.015, real_forwards.loc[all_rel.idxmax()].to_dict()


def test_real_forward_levels_sane(real_forwards):
    # AAPL traded ~175-190 in June 2023
    assert real_forwards["F"].between(150, 220).all()


def test_real_carry_sane(real_forwards):
    carry = implied_carry(real_forwards)
    assert len(carry) >= 3
    # dividend yield ~0.5%; allow generous band for EEP/noise in term slope
    assert carry["q_implied"].between(-0.05, 0.10).all(), carry.to_dict()
