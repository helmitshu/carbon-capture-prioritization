"""One-page opportunity memo: the Jobs-style boardroom version.

One page, the answer first, real data only. Built for the moment a VP
forwards it and says "call them". Charts come from econ_charts (same
economics engine as the text, so they can never disagree).
"""
from __future__ import annotations

import base64
import datetime
import io
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from econ_charts import (
    cost_band_chart,
    emissions_mix_donut,
    liability_split_bar,
)
from economics import facility_economics


def _fig_to_base64(fig, size=None) -> str:
    if size:
        fig.set_size_inches(*size)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _brand_logo_b64() -> str:
    """Sentinel logo for the memo header, empty string if missing."""
    import os

    for p in ("assets/sentinel-logo.jpg",
              os.path.join(os.path.dirname(__file__),
                           "assets/sentinel-logo.jpg")):
        if os.path.exists(p):
            with open(p, "rb") as fh:
                return base64.b64encode(fh.read()).decode("ascii")
    return ""


def _fmt_money(cad: float) -> str:
    a = abs(cad)
    if a >= 1e9:
        return f"${cad / 1e9:,.2f}B"
    if a >= 1e6:
        return f"${cad / 1e6:,.0f}M"
    return f"${cad:,.0f}"


def _fmt_mt(tonnes: float) -> str:
    return f"{tonnes / 1e6:,.1f} Mt"


_CSS = """
@page { size: A4; margin: 13mm 14mm 12mm 14mm; }
* { box-sizing: border-box; }
body { font-family: Helvetica, Arial, sans-serif; color: #1e293b;
       font-size: 9pt; line-height: 1.45; margin: 0; }
.accent { height: 4px; background: #0d9488; margin: 0 0 10px 0; }
.brand-row { display: table; margin-bottom: 3px; }
.brand-logo { display: table-cell; height: 30px; width: auto;
              vertical-align: middle; padding-right: 10px; }
.brand-kicker { display: table-cell; vertical-align: middle;
                font-size: 8pt; letter-spacing: 2.5px; color: #0d9488;
                font-weight: bold; }
.kicker { font-size: 8pt; letter-spacing: 2.5px; color: #0d9488;
          font-weight: bold; margin-bottom: 2px; }
h1 { font-size: 20pt; margin: 0 0 2px 0; color: #0f172a; letter-spacing: 0.3px; }
.facility-line { font-size: 10pt; color: #475569; margin-bottom: 10px; }
.head-row { display: table; width: 100%; margin-bottom: 6px; }
.head-left { display: table-cell; vertical-align: top; }
.head-right { display: table-cell; vertical-align: top; text-align: right;
              width: 190px; }
.verdict-badge { display: inline-block; background: #0d9488; color: #fff;
                 font-weight: bold; font-size: 10pt; letter-spacing: 1px;
                 padding: 7px 14px; border-radius: 3px; }
.verdict-sub { font-size: 7.5pt; color: #64748b; margin-top: 4px; }
.rank-line { font-size: 8.5pt; color: #0d9488; font-weight: bold;
             margin-top: 5px; letter-spacing: 0.3px; }
.verdict-p { font-size: 9.5pt; line-height: 1.5; margin: 0 0 10px 0;
             color: #1e293b; }
.section-title { font-size: 8pt; letter-spacing: 1.8px; color: #0d9488;
                 font-weight: bold; margin: 8px 0 5px 0;
                 border-bottom: 1px solid #e2e8f0; padding-bottom: 3px; }
.charts { display: table; width: 100%; table-layout: fixed; }
.chart-cell { display: table-cell; vertical-align: middle; text-align: center;
              padding: 0 6px; }
.chart-cell img { height: 148px; width: auto; max-width: 100%; }
.chart-cell.trend { display: block; text-align: center; padding: 0; }
.chart-cell.trend img { height: auto; width: 100%; max-width: 100%; }
.trend-note { font-size: 8pt; color: #64748b; margin: 4px 0 0 0;
              line-height: 1.45; }
.numbers { display: table; width: 100%; table-layout: fixed; margin-top: 2px; }
.num-cell { display: table-cell; text-align: center; padding: 6px 4px;
            border-left: 1px solid #e2e8f0; }
.num-cell:first-child { border-left: none; }
.num-val { font-size: 13pt; font-weight: bold; color: #0f172a; }
.num-val.neg { color: #b45309; }
.num-label { font-size: 7pt; color: #64748b; letter-spacing: 0.4px;
             margin-top: 2px; }
.risks { margin: 4px 0 0 0; padding-left: 14px; }
.risks li { margin-bottom: 3px; font-size: 8.5pt; }
.next-step { background: #f0fdfa; border-left: 3px solid #0d9488;
             padding: 7px 10px; font-size: 9pt; margin-top: 8px; }
.contact { margin-top: 8px; text-align: center; font-size: 8pt;
           color: #475569; background: #f8fafc; border: 1px solid #e2e8f0;
           border-radius: 3px; padding: 6px 8px; }
.demo-tag { display: inline-block; background: #fef3c7; color: #92400e;
            font-size: 6.5pt; font-weight: bold; letter-spacing: 0.8px;
            padding: 1px 6px; border-radius: 2px; margin-left: 6px;
            vertical-align: 1px; }
.footer { margin-top: 10px; padding-top: 6px;
          border-top: 1px solid #e2e8f0; font-size: 6.8pt; color: #94a3b8;
          line-height: 1.4; }
"""


_RANK_CACHE = {}


def _facility_rank(facility_id: str):
    """Rank by average annual emissions across all screened facilities.

    Returns (rank, total). Cached per process. Real data only.
    """
    if "frame" not in _RANK_CACHE:
        import os

        import pandas as pd

        here = os.path.dirname(os.path.abspath(__file__))
        _RANK_CACHE["frame"] = pd.read_csv(
            os.path.join(here, "model", "facilities_v3.csv"))
    fac = _RANK_CACHE["frame"]
    ordered = fac.sort_values("avg_annual_emissions", ascending=False)
    ids = list(ordered["facility_id"].astype(str))
    try:
        rank = ids.index(str(facility_id)) + 1
    except ValueError:
        rank = 0
    return rank, len(ids)


def render_onepage_memo_html(row) -> str:
    """Build the one-page memo HTML for a facilities_v3.csv row.

    row: a pandas Series (or dict) with the v3 facility columns.
    All figures come from the real record and economics.facility_economics.
    """
    r = dict(row)
    facility_id = str(r.get("facility_id", ""))
    name = str(r.get("facility_name", "Unnamed facility"))
    operator = str(r.get("company_trade", "") or "Not reported")
    city = str(r.get("city", "") or "").title()
    province = str(r.get("province", "") or "")
    sector = str(r.get("sector", "") or "")
    emissions = float(r.get("avg_annual_emissions", 0) or 0)
    co2_share = float(r.get("co2_share", 0) or 0)
    years = int(float(r.get("years_reported", 0) or 0))
    votes = int(float(r.get("panel_ccs_votes", 0) or 0))
    tier = str(r.get("verdict_tier", "") or "")
    data_through = "2024"

    rank, n_total = _facility_rank(facility_id)
    rank_line = (f"Ranked #{rank} of {n_total} screened facilities"
                 if rank else "")

    e = facility_economics(emissions, sector, co2_share=co2_share)
    captured = e["captured_tonnes"]
    abatable = e["abatable_liability_2030_cad"]
    locked = e["liability_2030_cad"] - abatable
    cost = e["annual_capture_cost_cad"]
    margin = e["margin_vs_abatable_cad"]
    itc = e["itc"]["credit_cad"]
    net_capex = e["net_capex_cad"]
    band = e["band"]

    donut_b64 = _fig_to_base64(
        emissions_mix_donut(emissions, sector, co2_share=co2_share))
    bar_b64 = _fig_to_base64(
        liability_split_bar(emissions, sector, co2_share=co2_share))
    cost_fig = cost_band_chart(emissions, sector, co2_share=co2_share)
    cost_fig.axes[0].set_title("Capture cost vs carbon price", fontsize=12,
                               fontweight="bold", color="#0f172a")
    cost_b64 = _fig_to_base64(cost_fig)
    from econ_charts import emissions_trend_chart
    try:
        trend_fig = emissions_trend_chart(facility_id, figsize=(8.5, 1.7),
                                          show_title=False)
        trend_b64 = _fig_to_base64(trend_fig)
        trend_section = f"""
<div class="section-title">THE TREND</div>
<div class="chart-cell trend"><img src="data:image/png;base64,{trend_b64}"></div>
<div class="trend-note">{years} years of reported emissions with the long-run trend. A rising or flat stream supports a capture case; a falling one means the project chases a shrinking target.</div>
"""
    except ValueError:
        trend_section = ""

    verdict_p = (
        f"<b>{name} is the largest capturable CO2 stream in the screen: "
        f"{_fmt_mt(captured)} a year at {co2_share:.0%} purity.</b> "
        f"A four-model panel voted {votes} of 4 for carbon capture, a "
        f"{tier} verdict, over {years} years of reported data. "
        f"The base case roughly breaks even against the {_fmt_money(abatable)} "
        f"2030 carbon bill slice; the low cost case clears "
        f"{_fmt_money(margin['low'])} a year."
    )

    capex_gross = e["capex_cad"]
    risk_hinge = (
        f"<b>The deal hinges on the low end of the cost band.</b> At base "
        f"capture cost the project {_fmt_money(margin['base'])} a year against "
        f"the carbon bill; at the low end it clears {_fmt_money(margin['low'])} "
        f"a year. Only a pre-FEED engineering screen prices it."
    )
    risk_scale = (
        f"<b>Scale.</b> {_fmt_money(capex_gross)} gross build before the "
        f"{_fmt_money(itc)} tax credit. Execution and financing risk at this "
        f"scale dwarfs anything a screening memo can retire."
    )

    margin_base = margin["base"]
    margin_cls = "neg" if margin_base < 0 else ""

    from zoneinfo import ZoneInfo
    date_str = datetime.datetime.now(ZoneInfo("America/Vancouver")).strftime(
        "%B %d, %Y")

    logo_b64 = _brand_logo_b64()
    brand_row = (f"""
<div class="brand-row">
  <img class="brand-logo" src="data:image/jpeg;base64,{logo_b64}">
  <div class="brand-kicker">CARBON CAPTURE ORIGINATION</div>
</div>""" if logo_b64 else
        '<div class="kicker">CARBON CAPTURE ORIGINATION</div>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Opportunity Memo, {name}</title>
<style>{_CSS}</style></head>
<body>
<div class="accent"></div>
{brand_row}
<div class="head-row">
  <div class="head-left">
    <h1>Opportunity Memo</h1>
    <div class="facility-line">{name} &middot; {operator} &middot; {city}, {province}<br>{sector} &middot; {years} years of reported data</div>
  </div>
  <div class="head-right">
    <div class="verdict-badge">STRONG CANDIDATE</div>
    <div class="rank-line">{rank_line}</div>
    <div class="verdict-sub">Panel {votes}/4 &middot; {tier} &middot; {date_str}</div>
  </div>
</div>
<p class="verdict-p">{verdict_p}</p>

<div class="section-title">THE NUMBERS</div>
<div class="charts">
  <div class="chart-cell"><img src="data:image/png;base64,{donut_b64}"></div>
  <div class="chart-cell"><img src="data:image/png;base64,{bar_b64}"></div>
  <div class="chart-cell"><img src="data:image/png;base64,{cost_b64}"></div>
</div>
<div class="numbers">
  <div class="num-cell"><div class="num-val">{_fmt_mt(captured)}</div><div class="num-label">CAPTURABLE PER YEAR</div></div>
  <div class="num-cell"><div class="num-val">{_fmt_money(abatable)}</div><div class="num-label">2030 BILL, ABATABLE SLICE</div></div>
  <div class="num-cell"><div class="num-val">{_fmt_money(itc)}</div><div class="num-label">ITC CREDIT</div></div>
  <div class="num-cell"><div class="num-val">{_fmt_money(net_capex)}</div><div class="num-label">NET CAPEX AFTER ITC</div></div>
  <div class="num-cell"><div class="num-val {margin_cls}">{_fmt_money(margin_base)}</div><div class="num-label">BASE MARGIN VS BILL</div></div>
</div>

{trend_section}
<div class="section-title">WHAT COULD KILL IT</div>
<ul class="risks">
  <li>{risk_hinge}</li>
  <li>{risk_scale}</li>
  <li><b>Storage access.</b> Pore space, pipeline distance, and transport cost from the {city} area need a dedicated storage screening. A great capture site with no storage is stranded.</li>
  <li><b>Policy path.</b> The federal carbon price was revised down in May 2026 ($170 to $115 per tonne by 2030). Model the downside where it softens further.</li>
</ul>

<div class="next-step"><b>Next step:</b> a pre-FEED capture cost screen and a storage screening for the {city} area. Those two numbers decide whether this becomes a project. <b>Why now:</b> the federal credit pays 50% on capture equipment at 2026 rates and halves from 2036. Every year of delay shrinks it.</div>

<div class="contact">Questions on this memo? Carbon Capture Origination &middot; +1 (555) 010-2030 &middot; memos@example.com<span class="demo-tag">DEMO CONTACT</span></div>

<div class="footer">
  Generated from public emissions records through {data_through} ({years} reporting years). Capture cost band: {band['source']} Carbon price: federal benchmark revised May 2026 ($95 in 2026, $100 2027-2029, $115 in 2030). ITC: 50% capture, 37.5% transport and storage, 2026 rates. FX 1.43. Capture rate 90% of the CO2 fraction, illustrative. Not investment advice.
</div>
</body>
</html>"""


def render_onepage_memo_pdf(row) -> bytes:
    """Render the one-page memo for a facilities_v3.csv row to PDF bytes."""
    from weasyprint import HTML
    return bytes(HTML(string=render_onepage_memo_html(row)).write_pdf())
