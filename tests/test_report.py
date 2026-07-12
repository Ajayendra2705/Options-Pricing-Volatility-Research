"""
Day 29 — report generation tests.

The report's whole claim is "no hand-typed numbers": every figure in report.html
and in the README block must trace to a results JSON.  So the tests check the
plumbing (numbers flow through, markers are respected, output is byte-stable)
rather than re-asserting the numbers themselves — those are already gated by the
per-day test modules.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import pytest

from src import report as rep

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
REAL_RESULTS = all((RESULTS_DIR / f"{n}.json").exists() for n in rep.RESULT_FILES)

pytestmark = pytest.mark.skipif(
    not REAL_RESULTS, reason="results JSONs absent — run `python main.py` first")


@pytest.fixture(scope="module")
def res() -> dict:
    return rep.load_results()


# ── formatters ───────────────────────────────────────────────────────────────

def test_formatters():
    assert rep.usd(1234.5) == "$1,234.50"
    assert rep.usd(-415.127) == "-$415.13"
    assert rep.usd(419.0, 0) == "$419"
    assert rep.pct(-0.015315) == "-1.53%"
    assert rep.num(-1.7008) == "-1.70"
    assert rep.volpts(0.0038944) == "0.39"          # fractional IV -> vol pts


def test_embed_png_missing_returns_none(tmp_path):
    assert rep.embed_png(tmp_path / "nope.png") is None


def test_embed_png_roundtrips_bytes(tmp_path):
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n fake bytes")
    uri = rep.embed_png(png)
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == png.read_bytes()


# ── load ─────────────────────────────────────────────────────────────────────

def test_load_results_reads_every_tracked_json(res):
    assert set(res) == set(rep.RESULT_FILES)
    assert res["metrics"]["capital_base_usd"] > 0


def test_run_report_raises_when_a_results_json_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rep, "RESULTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="missing results JSON"):
        rep.run_report()


# ── html: numbers come from the JSONs ────────────────────────────────────────

def test_html_headline_numbers_match_the_json(res):
    html = rep.build_html(res)
    m, costs = res["metrics"], res["costs_summary"]

    # each headline figure appears exactly as the JSON says it should
    assert rep.usd(costs["gross_pnl"]) in html
    assert rep.usd(costs["net_pnl"]) in html
    assert rep.pct(m["net_return_on_capital"]) in html
    assert rep.num(m["horizons"]["daily"]["sharpe"]) in html
    assert rep.num(m["statistical_honesty"]["sharpe"]["nw_tstat"]) in html
    assert rep.num(m["alpha_regression"]["beta"]) in html


def test_html_has_every_section_and_all_ten_positions(res):
    html = rep.build_html(res)
    for section in ("Headline", "volatility surface", "Signal", "attribution",
                    "Costs kill it", "Capital base", "Return distribution",
                    "Alpha isolation", "Statistical honesty", "Limitations"):
        assert section in html, f"missing section: {section}"

    # cost table carries one row per pre-registered position
    for pos in res["costs_summary"]["positions"]:
        assert f"{pos['date']} → {pos['expiry']}" in html


def test_html_is_self_contained(res):
    """No external fetches: plots are data URIs, css is inline, no <script>."""
    html = rep.build_html(res)
    assert "data:image/png;base64," in html
    assert "<script" not in html.lower()
    srcs = re.findall(r'src="([^"]+)"', html)
    assert srcs and all(s.startswith("data:") for s in srcs)
    assert not re.search(r'(href|src)="https?://', html)


def test_html_carries_no_timestamp(res):
    """Byte-stability rule: tracked outputs contain no clock-dependent text."""
    html = rep.build_html(res)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", html)   # ISO stamps
    assert rep.build_html(res) == html                             # deterministic


def test_html_escapes_json_prose(res):
    """Prose pulled from JSON is HTML-escaped, not injected raw."""
    doctored = json.loads(json.dumps(res))
    doctored["metrics"]["statistical_honesty"]["interpretation"] = "<b>x</b> & y"
    html = rep.build_html(doctored)
    assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in html
    assert "<b>x</b>" not in html


# ── readme block ─────────────────────────────────────────────────────────────

def test_readme_block_numbers_match_the_json(res):
    block = rep.build_readme_block(res)
    costs, m = res["costs_summary"], res["metrics"]
    assert rep.usd(costs["net_pnl"]) in block
    assert rep.pct(m["net_return_on_capital"]) in block
    assert rep.num(m["horizons"]["daily"]["sharpe"]) in block
    assert block.startswith(rep.README_START)
    assert block.endswith(rep.README_END)
    for plot in rep.README_PLOTS:
        assert plot in block


def test_update_readme_is_idempotent_and_preserves_prose(res, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Title\n\nintro prose\n\n{rep.README_START}\nSTALE\n{rep.README_END}\n\n"
        "## Hand-written section\n\nkept\n", encoding="utf-8")

    once = rep.update_readme(res, readme)
    assert "STALE" not in once
    assert "intro prose" in once and "## Hand-written section" in once
    assert rep.usd(res["costs_summary"]["net_pnl"]) in once

    twice = rep.update_readme(res, readme)
    assert twice == once            # regenerating does not drift or duplicate
    assert once.count(rep.README_START) == 1


def test_update_readme_refuses_without_markers(res, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\nno markers here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="markers"):
        rep.update_readme(res, readme)
    assert readme.read_text() == "# Title\n\nno markers here\n"   # untouched


# ── real deliverable ─────────────────────────────────────────────────────────

def test_report_html_deliverable_exists_and_is_current():
    """The tracked report.html reproduces bit-identically from current JSONs."""
    out = RESULTS_DIR / "report.html"
    assert out.exists(), "run `python main.py --stage report`"
    assert out.read_text(encoding="utf-8") == rep.build_html(rep.load_results())


def test_readme_auto_block_is_current():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    block = rep.build_readme_block(rep.load_results())
    assert block in text, "README AUTO:METRICS block is stale — rerun --stage report"
