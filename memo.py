"""One-click screening memo generation (v3 MVP).

Pure functions: no Streamlit, no I/O. The app calls render_memo_html for
a facility row and offers the PDF (or an HTML fallback) as a download.

Writing rules for all user-facing copy in this module: commas and
periods only, no em-dashes, plain phrasing, no personal names. The
client_name argument is the only name that ever appears, and only when
the caller passes one in.
"""
from __future__ import annotations

import ctypes.util
import datetime
import glob
import logging
import os

logger = logging.getLogger(__name__)

from assumptions import assumption_value

# Federal carbon price schedule, CAD per tonne CO2e.
# Centralized in assumptions.py, the single source of truth.
# Source: Federal benchmark revised May 15, 2026.
from assumptions import CARBON_PRICE_DECK

_MEMO_PRICE_2030 = CARBON_PRICE_DECK["schedule"][2030]  # 115 CAD/t
_MEMO_CAPTURE_RATE = assumption_value("capture_rate")  # illustrative capture rate
_MEMO_CAPEX_PER_TPA = assumption_value("capex_per_tpa_cad")  # illustrative CAD per tonne-per-annum build cost
_CAPTURE_RATE = _MEMO_CAPTURE_RATE
_CAPEX_PER_TPA = _MEMO_CAPEX_PER_TPA

_TIER_CLASSES = {"unanimous": "tier-unanimous",
                 "majority": "tier-majority",
                 "contested": "tier-contested"}

_CSS = """
  @page { size: A4; margin: 22mm 18mm 20mm 18mm; }
  * { box-sizing: border-box; }
  body { font-family: Helvetica, Arial, sans-serif; color: #1e293b; font-size: 10.5pt; line-height: 1.55; margin: 0; }
  .sample-strip { background: #b45309; color: #fff; font-size: 8.5pt; letter-spacing: 2px; text-align: center; padding: 6px 0; font-weight: bold; }
  .cover { text-align: center; padding: 60px 30px 30px 30px; }
  .cover .kicker { color: #0f766e; font-weight: bold; letter-spacing: 3px; font-size: 10pt; }
  .cover h1 { font-size: 30pt; margin: 14px 0 6px 0; color: #0f172a; }
  .cover .sub { font-size: 13pt; color: #475569; }
  .cover .meta { margin-top: 44px; font-size: 10pt; color: #475569; text-align: left; display: inline-block; }
  .cover .meta td { padding: 4px 14px 4px 0; }
  .cover .meta td:first-child { color: #0f766e; font-weight: bold; }
  .disclaimer { margin: 40px auto 0 auto; max-width: 520px; font-size: 8.5pt; color: #64748b; border: 1px solid #cbd5e1; padding: 10px 14px; border-radius: 6px; text-align: left; }
  h2 { color: #0f172a; font-size: 16pt; border-bottom: 3px solid #0f766e; padding-bottom: 6px; margin-top: 34px; }
  h3 { color: #0f766e; font-size: 12pt; margin-top: 22px; margin-bottom: 6px; }
  .pagebreak { page-break-before: always; }
  .verdict { background: #ecfdf5; border: 2px solid #0f766e; border-radius: 8px; padding: 16px 20px; margin: 16px 0; }
  .verdict .grade { font-size: 20pt; font-weight: bold; color: #0f766e; }
  .tier-badge { display: inline-block; font-weight: bold; font-size: 10pt; border-radius: 6px; padding: 4px 12px; margin: 10px 0 0 0; }
  .tier-unanimous { background: #ecfdf5; color: #0f766e; border: 2px solid #0f766e; }
  .tier-majority { background: #fffbeb; color: #b45309; border: 2px solid #b45309; }
  .tier-contested { background: #fef2f2; color: #b91c1c; border: 2px solid #b91c1c; }
  .grid { display: table; width: 100%; border-collapse: collapse; margin: 12px 0; }
  .grid .row { display: table-row; }
  .grid .cell { display: table-cell; width: 50%; padding: 10px 14px; border: 1px solid #e2e8f0; vertical-align: top; }
  .grid .cell .label { font-size: 8.5pt; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
  .grid .cell .value { font-size: 14pt; font-weight: bold; color: #0f172a; }
  .steps { counter-reset: step; margin: 14px 0; padding: 0; }
  .steps li { list-style: none; margin: 0 0 12px 0; padding: 12px 14px 12px 54px; background: #f8fafc; border-left: 4px solid #0f766e; border-radius: 0 6px 6px 0; position: relative; }
  .steps li::before { counter-increment: step; content: counter(step); position: absolute; left: 12px; top: 10px; width: 28px; height: 28px; background: #0f766e; color: #fff; border-radius: 50%; text-align: center; line-height: 28px; font-weight: bold; font-size: 11pt; }
  .steps li b { color: #0f172a; }
  table.econ { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 10pt; }
  table.econ th { background: #0f766e; color: #fff; text-align: left; padding: 8px 10px; }
  table.econ td { padding: 8px 10px; border-bottom: 1px solid #e2e8f0; }
  table.econ tr:last-child td { border-bottom: 2px solid #0f766e; font-weight: bold; }
  .note { font-size: 8.5pt; color: #64748b; font-style: italic; }
  .risk { background: #fffbeb; border: 1px solid #f59e0b; border-radius: 6px; padding: 10px 14px; margin: 8px 0; }
  .risk b { color: #92400e; }
  .cta { background: #0f172a; color: #fff; border-radius: 8px; padding: 18px 22px; margin-top: 20px; }
  .cta b { color: #5eead4; }
  ul.tight li { margin-bottom: 6px; }
  .footer { margin-top: 30px; font-size: 8pt; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 8px; }
"""


def _s(value) -> str:
    """Clean a value for user-facing copy: no em/en dashes, ever."""
    return (str(value).replace("\u2014", ",").replace("\u2013", ",").strip())


def _fmt_mt(tonnes: float) -> str:
    return f"{tonnes / 1e6:.2f} Mt"


def _fmt_money(cad: float) -> str:
    if cad >= 1e9:
        return f"${cad / 1e9:.2f}B"
    if cad >= 1e6:
        return f"${cad / 1e6:,.0f}M"
    return f"${cad / 1e3:,.0f}k"


def _tier_badge_html(tier: str, panel_votes: int) -> str:
    cls = _TIER_CLASSES.get(tier, "")
    if not cls:
        return ""
    return (f'<div><span class="tier-badge {cls}">Panel: {panel_votes}/4, '
            f'{_s(tier)}</span></div>')


def _econ_section_legacy(emissions: float, data_through: str,
                         deck: dict, deck_line: str) -> str:
    """Original illustrative economics table. v2 behavior, frozen."""
    liability = emissions * _MEMO_PRICE_2030
    captured = emissions * _CAPTURE_RATE
    capex = captured * _CAPEX_PER_TPA
    sens_rows = ""
    for rate in (60, 90, 120):
        cost = captured * rate
        margin = liability - cost
        sens_rows += (f"<tr><td>${rate} per tonne</td>"
                      f"<td>{_fmt_money(cost)}</td>"
                      f"<td>{'+' if margin >= 0 else ''}"
                      f"{_fmt_money(margin)}</td></tr>")
    return f"""<h2>4. Illustrative economics envelope</h2>
<p>This section shows the shape of the investment question. Labeled figures are illustrative placeholders showing the format. A paid memo runs your commercial assumptions.</p>
<table class="econ">
  <tr><th>Line</th><th>Figure</th><th>Basis</th></tr>
  <tr><td>Annual CO2 available ({data_through})</td><td>{_fmt_mt(emissions)}</td><td>Reported</td></tr>
  <tr><td>Illustrative capture rate</td><td>90%</td><td>Illustrative</td></tr>
  <tr><td>Illustrative captured volume</td><td>{_fmt_mt(captured)} per year</td><td>Calculated</td></tr>
  <tr><td>Illustrative capex envelope</td><td>~{_fmt_money(capex)}</td><td>Illustrative, at industry average build cost</td></tr>
  <tr><td>Carbon liability at ${_MEMO_PRICE_2030}/t (2030 revised schedule)</td><td>~{_fmt_money(liability)} per year</td><td>{_fmt_mt(emissions)} at revised 2030 price. Source: {deck["source"]}.</td></tr>
</table>
<p>The federal carbon price was revised in May 2026: {deck_line}. Against that, the question is the levelized cost of capture at this site:</p>
<table class="econ">
  <tr><th>Illustrative capture cost</th><th>Annual cost on {_fmt_mt(captured)}</th><th>Margin vs {_fmt_money(liability)} liability</th></tr>
{sens_rows}
</table>
<p class="note">Capture costs are illustrative placeholders. Site engineering sets the real number, which is exactly what the recommended next steps price out.</p>"""


def _econ_section_v31(emissions: float, sector: str, co2_share: float,
                      data_through: str, deck: dict, deck_line: str) -> str:
    """Sourced economics envelope for v3.1.

    Sector capture cost bands and the federal CCUS investment tax
    credit replace the flat illustrative table. Capture volumes scale
    by the facility CO2 share, and the margin is measured against the
    abatable slice of the carbon liability. Deferred import keeps
    economics.py (which imports this module) free of a cycle.
    """
    from economics import facility_economics
    econ = facility_economics(emissions, sector, co2_share=co2_share)
    band = econ["band"]
    captured = econ["captured_tonnes"]
    liability = econ["liability_2030_cad"]
    abatable = econ["abatable_liability_2030_cad"]
    itc = econ["itc"]
    ts_low, ts_high = econ["ts_adder_cad"]
    sens_rows = ""
    for level, label in (("low", "Low case"), ("base", "Base case"),
                         ("high", "High case")):
        cost = econ["annual_capture_cost_cad"][level]
        margin = econ["margin_vs_abatable_cad"][level]
        sens_rows += (f"<tr><td>{label}, "
                      f"${band[f'{level}_cad']:,.0f} per tonne</td>"
                      f"<td>{_fmt_money(cost)}</td>"
                      f"<td>{'+' if margin >= 0 else ''}"
                      f"{_fmt_money(margin)}</td></tr>")
    assumptions = " ".join(_s(a) for a in econ["assumptions"])
    return f"""<h2>4. Indicative economics envelope</h2>
<p>This section puts sourced public benchmarks around the investment question. Every assumption is labeled. A paid memo runs your commercial assumptions and current data.</p>
<table class="econ">
  <tr><th>Line</th><th>Figure</th><th>Basis</th></tr>
  <tr><td>Annual CO2e available ({data_through})</td><td>{_fmt_mt(emissions)}</td><td>Reported</td></tr>
  <tr><td>CO2 share of emissions</td><td>{co2_share:.0%}</td><td>Reported</td></tr>
  <tr><td>Indicative capture rate of the CO2 fraction</td><td>{assumption_value("capture_rate"):.0%}</td><td>Illustrative</td></tr>
  <tr><td>Indicative captured volume</td><td>{_fmt_mt(captured)} per year</td><td>Calculated</td></tr>
  <tr><td>Sector capture cost band</td><td>${band["low_cad"]:,.0f} to ${band["high_cad"]:,.0f} per tonne</td><td>{_s(band["source"])}</td></tr>
  <tr><td>Transport and storage adder</td><td>${ts_low:,.0f} to ${ts_high:,.0f} per tonne</td><td>Global CCS Institute, pipeline plus storage.</td></tr>
  <tr><td>Indicative capex envelope</td><td>~{_fmt_money(econ["capex_cad"])}</td><td>Illustrative, at industry average build cost</td></tr>
  <tr><td>Federal CCUS investment tax credit</td><td>~{_fmt_money(itc["credit_cad"])} refundable</td><td>{itc["capture_rate"] * 100:g}% capture equipment, {itc["tsu_rate"] * 100:g}% transport storage and use. Source: IEA policy tracker, June 2026.</td></tr>
  <tr><td>Net capex after credit</td><td>~{_fmt_money(econ["net_capex_cad"])}</td><td>Calculated</td></tr>
  <tr><td>Carbon liability at ${_MEMO_PRICE_2030}/t (2030 revised schedule)</td><td>~{_fmt_money(liability)} per year</td><td>{_fmt_mt(emissions)} at revised 2030 price. Source: {deck["source"]}.</td></tr>
  <tr><td>Abatable slice of the liability ({co2_share:.0%} CO2)</td><td>~{_fmt_money(abatable)} per year</td><td>Calculated. Capture can only address the CO2 fraction.</td></tr>
</table>
<p>The federal carbon price was revised in May 2026: {deck_line}. Against that, the levelized capture cost at this site decides the project:</p>
<table class="econ">
  <tr><th>Capture cost case</th><th>Annual cost on {_fmt_mt(captured)}</th><th>Margin vs {_fmt_money(abatable)} abatable liability</th></tr>
{sens_rows}
</table>
<p class="note">Capture cost bands are public benchmarks for the sector, not site engineering. The recommended next steps price the real number.</p>
<p class="note">Assumptions: {assumptions}</p>"""


def render_memo_html(facility: dict, reasons: list[str], tier: str,
                     panel_votes: int, client_name: str = "",
                     economics: str = "legacy") -> str:
    """Render a screening memo as an HTML string.

    facility keys used (all optional, sensible fallbacks): facility_name,
    city, province, operator, sector, avg_annual_emissions (tonnes),
    co2_share (0-1), years_reported, priority, model_prediction,
    data_through. reasons is the panel reason list (may be empty).
    tier is one of unanimous/majority/contested (or "" to omit).
    client_name="" keeps the SAMPLE banner; a name removes it.
    economics="legacy" keeps the original illustrative economics table
    (v2 behavior, frozen). economics="v31" uses the sourced sector
    cost bands and CCUS tax credit from economics.py (v3 only).
    """
    f = {k: _s(v) for k, v in facility.items()}
    name = f.get("facility_name", "Unnamed facility")
    city = f.get("city", "Not reported")
    province = f.get("province", "Not reported")
    operator = f.get("operator", "Not reported")
    sector = f.get("sector", "Not reported")
    data_through = f.get("data_through", "2023")
    try:
        emissions = float(facility.get("avg_annual_emissions", 0) or 0)
    except (TypeError, ValueError):
        emissions = 0.0
    try:
        co2_share = float(facility.get("co2_share", 0) or 0)
    except (TypeError, ValueError):
        co2_share = 0.0
    try:
        years = int(float(facility.get("years_reported", 0) or 0))
    except (TypeError, ValueError):
        years = 0

    verdict = f.get("model_prediction") or f.get("priority", "")
    is_ccs = verdict == "CCS Candidate"
    grade = "STRONG CANDIDATE" if is_ccs else "UTILIZATION CANDIDATE"
    tier = _s(tier)
    badge = _tier_badge_html(tier, panel_votes)
    tier_sentence = (f" A panel of four models voted {panel_votes} of 4 for "
                     f"capture, a {tier} verdict." if badge else "")
    if is_ccs:
        summary = (f"The {name} emits roughly {_fmt_mt(emissions)} of CO2 "
                   f"per year, of which {co2_share:.1%} is CO2, across "
                   f"{years} years of reported data. The screening model "
                   f"rates it a strong candidate for carbon capture."
                   f"{tier_sentence} The open questions are capture cost, "
                   f"confirmed by engineering, and storage access from this "
                   f"location.")
    else:
        summary = (f"The {name} emits roughly {_fmt_mt(emissions)} per year, "
                   f"of which {co2_share:.1%} is CO2, across {years} years "
                   f"of reported data. The screening model rates it a "
                   f"better fit for carbon utilization than for capture and "
                   f"storage.{tier_sentence} Treat this as a steer toward "
                   f"utilization pathways, not a capture project.")

    clean_reasons = [_s(r) for r in reasons if _s(r)]
    if clean_reasons:
        reasons_html = "".join(f"<li>{r}</li>" for r in clean_reasons)
    else:
        reasons_html = ("<li>No panel reasons were recorded for this "
                        "facility.</li>")

    deck = CARBON_PRICE_DECK
    sched = deck["schedule"]
    deck_line = (f"${sched[2026]} per tonne in 2026, ${sched[2027]} per "
                 f"tonne from 2027 through 2029, rising to "
                 f"${sched[2030]} per tonne in 2030")

    econ_section = _econ_section_legacy(
        emissions, data_through, deck, deck_line)
    if economics == "v31":
        econ_section = _econ_section_v31(
            emissions, sector, co2_share, data_through, deck, deck_line)

    date_str = datetime.date.today().strftime("%B %Y")
    if client_name and _s(client_name):
        strip = ""
        prepared_for = _s(client_name)
        classification = f"Prepared for {_s(client_name)}."
    else:
        strip = ('<div class="sample-strip">SAMPLE DOCUMENT, ILLUSTRATIVE '
                 'FIGURES, NOT INVESTMENT ADVICE</div>')
        prepared_for = "Sample client (not a real engagement)"
        classification = "Sample. Illustrative figures throughout."

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Screening Memo, {_s(name)}</title>
<style>{_CSS}</style>
</head>
<body>
{strip}
<div class="cover">
  <div class="kicker">CARBON CAPTURE ORIGINATION</div>
  <h1>Opportunity Screening Memo</h1>
  <div class="sub">Facility level diligence for carbon capture project developers</div>
  <table class="meta">
    <tr><td>Prepared for</td><td>{prepared_for}</td></tr>
    <tr><td>Subject facility</td><td>{name}, {city}, {province}</td></tr>
    <tr><td>Operator on record</td><td>{operator}</td></tr>
    <tr><td>Date</td><td>{date_str}</td></tr>
    <tr><td>Classification</td><td>{classification}</td></tr>
  </table>
  <div class="disclaimer">
    This memo was generated from public emissions records. Figures marked illustrative show the format of the analysis. A paid memo covers your target facilities with current data and your commercial assumptions. Nothing here is investment advice.
  </div>
</div>

<div class="pagebreak"></div>
{strip}
<h2>What you are buying</h2>
<p>One memo per facility. Five business days. Each memo answers a single question your team asks before spending real diligence money: <b>does this facility deserve a closer look for carbon capture, and what would kill the deal?</b></p>
<p>The memo fuses public emissions records, sector screening, and an illustrative economics envelope into a plain language verdict with the risks stated up front. It is the document that sits between your origination longlist and your first engineering dollar.</p>
<h3>How your team uses it</h3>
<ol class="steps">
  <li><b>Screen.</b> You bring a longlist of facilities, or we screen the province for you. Every facility above the emissions threshold gets scored.</li>
  <li><b>Shortlist.</b> You receive one memo per facility. The verdict and the risk list tell you which ten deserve attention and which forty do not.</li>
  <li><b>Deep dive.</b> Your engineers and commercial team spend their FEED budget only on the shortlist, armed with the open questions each memo flagged.</li>
  <li><b>Decide.</b> The memos compound into a pipeline view: every screened facility, its verdict, its risks, in one place your investment committee can read.</li>
</ol>
<h3>How it pays for itself</h3>
<p>A full front end engineering design study runs into seven figures and the better part of a year. This memo costs a fraction of one percent of that and arrives in five days. It pays for itself the first time it stops you from engineering the wrong facility, or points your engineers at the right one three months earlier.</p>
<h3>What it is not</h3>
<p>It is not an engineering study, it does not replace site specific design, and it does not predict policy. Every assumption is labeled. Every open question is listed. If a facility is a bad fit, the memo says so, because a screening tool that never says no is worthless.</p>

<div class="pagebreak"></div>
{strip}
<h2>1. Executive summary</h2>
<div class="verdict">
  <div class="grade">{grade}</div>
  {badge}
  <p style="margin:8px 0 0 0;">{summary}</p>
</div>

<h2>2. Facility profile</h2>
<div class="grid">
  <div class="row">
    <div class="cell"><div class="label">Average annual emissions</div><div class="value">{_fmt_mt(emissions)} CO2e</div></div>
    <div class="cell"><div class="label">CO2 share of total</div><div class="value">{co2_share:.1%}</div></div>
  </div>
  <div class="row">
    <div class="cell"><div class="label">Reporting history</div><div class="value">{years} years</div></div>
    <div class="cell"><div class="label">Sector</div><div class="value" style="font-size:11pt;">{sector}</div></div>
  </div>
  <div class="row">
    <div class="cell"><div class="label">Province</div><div class="value" style="font-size:11pt;">{province}</div></div>
    <div class="cell"><div class="label">Operator on record</div><div class="value" style="font-size:11pt;">{operator}</div></div>
  </div>
</div>
<p>{name} operates in {sector} in {province}. The record covers {years} reporting years through {data_through}.</p>

<div class="pagebreak"></div>
{strip}
<h2>3. Why it qualifies</h2>
<p>The screening model flags this facility as a {_s(verdict).lower() or "screened facility"}. In plain language:</p>
<ul class="tight">
{reasons_html}
</ul>

{econ_section}

<div class="pagebreak"></div>
{strip}
<h2>5. Risks and open questions</h2>
<p>The part most screening tools leave out. Each of these is a question to answer before engineering dollars move.</p>
<div class="risk"><b>Capture cost is unconfirmed.</b> The economics above turn entirely on the per tonne capture cost at this specific plant. Only a pre FEED engineering screen prices it. Treat every figure in section 4 as a hypothesis until then.</div>
<div class="risk"><b>Storage access.</b> Suitable pore space, pipeline distance, and transport cost need a dedicated storage screening for the {city} area. A great capture site with no storage is a stranded asset.</div>
<div class="risk"><b>Plant operating life.</b> {years} years of history is strong, but capture investments amortize over decades. Confirm remaining operating life and planned investments with the operator.</div>
<div class="risk"><b>Policy path.</b> The carbon price was already revised down once, in May 2026, from $170 to $115 per tonne by 2030. Model the downside case where the path softens further, and the upside case where it firms.</div>
<div class="risk"><b>Data vintage.</b> Public records in this memo run to {data_through}. The paid memo refreshes to the latest reporting year and flags any material change.</div>

<h2>6. Recommended next steps</h2>
<ol class="steps">
  <li><b>Pre FEED capture screen.</b> Engage engineering for a site specific capture cost band. This single number decides the project.</li>
  <li><b>Storage screening.</b> Basin assessment, injectivity, distance and transport cost from {city}.</li>
  <li><b>Operator engagement.</b> Confirm plant life, investment plans, and appetite for a capture integration partnership.</li>
  <li><b>Commercial structure.</b> Map the federal CCUS investment tax credit and carbon price exposure against the cost band before committing to FEED.</li>
</ol>

<div class="cta">
  <b>What happens next.</b> This memo covers one facility. A pilot engagement screens your full longlist the same way, one memo per facility. The question it answers is always the same: where should the engineering dollars go first.
</div>

<div class="footer">
  Methodology: facility screening over Canada Greenhouse Gas Reporting Program records, threshold and sector rules applied at facility level. Carbon price: {deck["source"]}. Illustrative figures are placeholders. Not investment advice.
</div>

</body>
</html>"""


def render_memo_pdf(html: str) -> bytes:
    """Render memo HTML to PDF bytes.

    Raises ImportError("weasyprint not installed") when WeasyPrint is
    unavailable or its system libraries are missing, so callers can
    fall back to the HTML download.
    """
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as e:
        logger.warning(
            "weasyprint unavailable, HTML fallback: %r | so_exists=%s "
            "find_library=%s glob=%s",
            e,
            os.path.exists("/usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0"),
            ctypes.util.find_library("gobject-2.0"),
            glob.glob("/usr/lib/*/libgobject*")
            + glob.glob("/lib/*/libgobject*"),
        )
        raise ImportError("weasyprint not installed")
    return bytes(HTML(string=html).write_pdf())
    return bytes(HTML(string=html).write_pdf())
