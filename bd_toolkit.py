"""BD Toolkit: origination modules for carbon capture developers.

Screening tells you which facilities matter. This module turns that
into account level action. Who owns them, whether capture is buildable
there, what the economics look like under your assumptions, what to
say on the call, and what changed since you last looked.

Pure logic is Streamlit free at import time, so tests can import this
module without a Streamlit runtime. The render_* functions import
Streamlit lazily inside the function body.

Writing rules for user facing copy: commas and periods only, plain
phrasing, no personal names, no emojis. Estimates are labeled.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

from economics import carbon_liability_cad, facility_economics

_HERE = Path(__file__).resolve().parent

INDEPENDENT_OPERATOR = "Independent operator"

_BLANK_COMPANIES = {
    "", "not applicable", "n/a", "na", "unknown", "none", "tbd",
    "not available", "private",
}

# City name variants mapped to the canonical key in data/city_coords.csv.
# Only mapped when the anchor place is named in the string. Anything
# else stays unmatched and scores proximity as unknown, never guessed.
CITY_ALIASES = {
    "FT MCMURRAY": "FORT MCMURRAY",
    "22 KM NE OF FORT MCMURRAY": "FORT MCMURRAY",
    "RM OF WOOD BUFFALO": "FORT MCMURRAY",
    "REGIONAL MUNICIPALITY OF WOOD BUFFALO": "FORT MCMURRAY",
    "WOOD BUFFALO": "FORT MCMURRAY",
    "GRAND PRAIRIE": "GRANDE PRAIRIE",
    "CONKIN": "CONKLIN",
    "COUNTY OF ATHABASCA": "ATHABASCA",
    "MD OF BONNYVILLE": "BONNYVILLE",
}

# Readiness proximity bands: (max km, points).
_PROX_BANDS = ((80, 30), (150, 20), (300, 10))

# Scenario verdict thresholds on margin in CAD.
_MARGINAL_FLOOR_CAD = -50_000_000


def default_path(name: str) -> str:
    """Resolve a data file under the repo data directory."""
    return str(_HERE / "data" / name)


def fmt_money_cad(x) -> str:
    """Compact CAD money label, for example $1.2B or $340M."""
    x = float(x or 0)
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e9:
        return f"{sign}${a / 1e9:.1f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.0f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:.0f}K"
    return f"{sign}${a:,.0f}"


def fmt_mt(tonnes) -> str:
    """Megatonne label, for example 12.1 Mt."""
    return f"{float(tonnes or 0) / 1e6:.1f} Mt"


def normalize_company(name) -> str:
    """Map blank or placeholder company names to Independent operator."""
    s = str(name or "").strip()
    if s.lower() in _BLANK_COMPANIES:
        return INDEPENDENT_OPERATOR
    return s


@lru_cache(maxsize=4096)
def _econ_cached(emissions: float, sector: str,
                 co2_share: float) -> dict:
    """Cached facility_economics for aggregation loops."""
    return facility_economics(float(emissions), str(sector),
                             co2_share=float(co2_share))


def _econ_key(emissions, sector, co2_share):
    return (round(float(emissions or 0), 2), str(sector or ""),
            round(float(co2_share or 0), 4))


def owner_portfolios(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate facilities to account level by company.

    Returns one row per company: facility count, total average annual
    emissions, total 2030 carbon liability, total ITC credit, and the
    tier mix. Sorted by liability, highest first.
    """
    rows = []
    work = df.copy()
    work["_company"] = work["company_trade"].apply(normalize_company)
    for company, g in work.groupby("_company"):
        liability = 0.0
        itc = 0.0
        for _, r in g.iterrows():
            liability += carbon_liability_cad(r["avg_annual_emissions"],
                                              2030)
            econ = _econ_cached(*_econ_key(r["avg_annual_emissions"],
                                          r["sector"], r["co2_share"]))
            itc += econ["itc"]["credit_cad"]
        tiers = g["verdict_tier"].value_counts() if "verdict_tier" in g else {}
        rows.append({
            "company": company,
            "n_facilities": int(len(g)),
            "total_emissions_t": float(g["avg_annual_emissions"].sum()),
            "total_liability_2030_cad": round(liability, 2),
            "total_itc_cad": round(itc, 2),
            "tier_unanimous": int(tiers.get("unanimous", 0)),
            "tier_majority": int(tiers.get("majority", 0)),
            "tier_contested": int(tiers.get("contested", 0)),
            "provinces": sorted({str(p) for p in g["province"].unique()})
            if "province" in g else [],
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("total_liability_2030_cad",
                             ascending=False).reset_index(drop=True)
    return out


def why_this_account(owner_row: pd.Series) -> list:
    """Auto generated bullets that make the account case in plain words."""
    n = int(owner_row["n_facilities"])
    u = int(owner_row["tier_unanimous"])
    bullets = [
        f"{u} of {n} facilities carry a unanimous verdict.",
        ("One conversation covers about "
         f"{fmt_money_cad(owner_row['total_liability_2030_cad'])} "
         "in 2030 liability."),
        ("Eligible ITC across the account is about "
         f"{fmt_money_cad(owner_row['total_itc_cad'])}."),
    ]
    provs = owner_row["provinces"]
    if len(provs) > 1:
        bullets.append(f"Facilities span {len(provs)} provinces. "
                       "One owner, many options.")
    if u == n and n > 1:
        bullets.append("Every site in the account is a consensus pick. "
                       "This is a platform conversation, not a single asset.")
    return bullets


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great circle distance in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def load_hubs(path: str | None = None) -> list:
    """Load hub points. Each entry carries its public source."""
    path = path or default_path("hubs.csv")
    df = pd.read_csv(path)
    return df.to_dict("records")


def load_city_coords(path: str | None = None) -> dict:
    """Load city coordinates keyed by canonical uppercase name."""
    path = path or default_path("city_coords.csv")
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        out[str(r["city_key"]).strip().upper()] = {
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "note": str(r.get("note") or "").strip(),
            "source": str(r.get("source") or "").strip(),
        }
    return out


def canonical_city(city) -> str | None:
    """Canonical lookup key for a dataset city name, or None."""
    s = str(city or "").strip().upper()
    if not s or s in {"UNKNOWN", "N/A", "NA"}:
        return None
    return CITY_ALIASES.get(s, s)


def readiness_score(avg_annual_emissions, co2_share, city,
                    hubs: list, city_coords: dict) -> dict:
    """Transparent 0 to 100 readiness score for one facility.

    Scale is worth 40 points, four points per megatonne of average
    annual emissions, capped at 40. Purity is worth 30 points, the
    CO2 share times 30. Proximity is worth 30 points by distance to
    the nearest hub in data/hubs.csv. Distances are estimates from
    public approximate hub locations. Cities we cannot place score
    out of 70, with a visible note.
    """
    emissions = float(avg_annual_emissions or 0)
    share = min(max(float(co2_share or 0), 0.0), 1.0)
    scale_pts = min(40.0, (emissions / 1e6) * 4.0)
    purity_pts = share * 30.0

    key = canonical_city(city)
    loc = city_coords.get(key) if key else None
    prox_pts = None
    prox_km = None
    nearest = None
    note = ""
    if loc is None:
        note = ("City is not in the coordinate lookup, so proximity is "
                "unknown. Scored out of 70 from scale and purity only.")
        total = scale_pts + purity_pts
        denom = 70
    else:
        best = None
        for h in hubs:
            d = haversine_km(loc["lat"], loc["lon"],
                             float(h["lat"]), float(h["lon"]))
            if best is None or d < best[0]:
                best = (d, h)
        prox_km, hub = best
        nearest = str(hub["name"])
        prox_pts = 0.0
        for max_km, pts in _PROX_BANDS:
            if prox_km < max_km:
                prox_pts = float(pts)
                break
        note = ("Nearest hub is an estimate from public approximate "
                "locations.")
        total = scale_pts + purity_pts + prox_pts
        denom = 100
    return {
        "total": round(total, 1),
        "denominator": denom,
        "scale_pts": round(scale_pts, 1),
        "purity_pts": round(purity_pts, 1),
        "prox_pts": (round(prox_pts, 1) if prox_pts is not None else None),
        "prox_km": (round(prox_km, 1) if prox_km is not None else None),
        "nearest_hub": nearest,
        "city_key": key,
        "note": note,
    }


def scenario_row(avg_annual_emissions, sector, co2_share, price: float,
                 itc_rate_pct: float, cost_mult: float) -> dict:
    """Recompute per facility economics under scenario assumptions.

    Liability and the abatable slice scale with the carbon price
    against the 2030 federal benchmark. Capture cost is implied
    honestly from the economics module: abatable value minus the
    base margin, then scaled by the cost multiplier. The custom ITC
    rate applies to the modeled capex split, keeping the capture and
    transport rate structure from economics.itc_credit_cad.
    """
    emissions = float(avg_annual_emissions or 0)
    share = min(max(float(co2_share or 0), 0.0), 1.0)
    base = _econ_cached(*_econ_key(emissions, sector, share))

    liability = carbon_liability_cad(emissions, 2030) * (price / 115.0)
    abatable = (base["abatable_liability_2030_cad"] * (price / 115.0))
    implied_cost = (base["abatable_liability_2030_cad"]
                    - base["margin_vs_abatable_cad"]["base"])
    margin = abatable - implied_cost * cost_mult

    itc_base = base["itc"]
    cap_rate = float(itc_base["capture_rate"])
    tsu_rate = float(itc_base["tsu_rate"])
    ratio = (tsu_rate / cap_rate) if cap_rate > 0 else 0.0
    r = float(itc_rate_pct) / 100.0
    itc_credit = (base["capex_capture_cad"] * r
                  + base["capex_tsu_cad"] * r * ratio)

    if margin > 0:
        verdict = "Economic"
    elif margin >= _MARGINAL_FLOOR_CAD:
        verdict = "Marginal"
    else:
        verdict = "Underwater"
    return {
        "liability_cad": round(liability, 2),
        "abatable_cad": round(abatable, 2),
        "itc_credit_cad": round(itc_credit, 2),
        "margin_cad": round(margin, 2),
        "verdict": verdict,
        "method": ("Capture cost is implied as abatable value minus base "
                   "margin from economics.facility_economics, then scaled. "
                   "ITC at the custom rate keeps the modeled capex split "
                   "and rate structure."),
    }


# ---------------------------------------------------------------------------
# Outreach kit
# ---------------------------------------------------------------------------

def outreach_brief(facility: dict, readiness: dict) -> dict:
    """Build a one page brief and a short outreach email for a facility.

    The technical line is purity conditional. Only a CO2 share of 0.99
    or higher earns the well suited claim.
    """
    name = str(facility.get("facility_name", ""))
    owner = normalize_company(facility.get("company_trade", ""))
    emissions = float(facility.get("avg_annual_emissions") or 0)
    share = float(facility.get("co2_share") or 0)
    liability = carbon_liability_cad(emissions, 2030)
    econ = _econ_cached(*_econ_key(emissions, facility.get("sector", ""),
                                  share))
    itc = econ["itc"]["credit_cad"]
    tier = str(facility.get("verdict_tier", ""))

    hooks = [
        (f"This site faces about {fmt_money_cad(liability)} a year in "
         "carbon exposure by 2030. That number opens the budget "
         "conversation."),
        (f"The federal credit could cover roughly {fmt_money_cad(itc)} "
         "of a capture investment. The rate steps down after 2035, so "
         "timing matters."),
    ]
    if tier == "unanimous":
        hooks.append("Every model agrees on this site, which means "
                     "competitors see it too. The first call wins.")
    elif tier == "contested":
        hooks.append("Contested verdicts are where whitespace lives. "
                     "Fewer bidders, more room to shape the deal.")
    else:
        flag = "scale" if readiness["scale_pts"] >= readiness["purity_pts"] \
            else "gas purity"
        hooks.append(f"Our screen flags this site on {flag}. That is the "
                     "technical opener for the first call.")

    talking_points = [
        ("We screen every large emitter in the country, and this site "
         "stood out on the numbers."),
        (f"The 2030 math is about {fmt_money_cad(liability)} of exposure "
         f"against roughly {fmt_money_cad(itc)} of credit support."),
        ("Twenty minutes is enough to see whether the technical case "
         "holds."),
    ]

    if share >= 0.99:
        tech_line = ("The gas stream is well suited to capture, so this is "
                     "one of the stronger technical cases we see.")
    else:
        tech_line = ("Our screening flags this site as worth a technical "
                     "look, starting with the gas stream.")

    email_subject = f"A thought on {name} and the 2030 carbon bill"
    email_body = "\n\n".join([
        ("Hi, we have been tracking large emitters across Canada and "
         f"{name} stood out."),
        ("At the scheduled carbon price the site faces about "
         f"{fmt_money_cad(liability)} a year in exposure by 2030."),
        ("The federal credit could cover roughly "
         f"{fmt_money_cad(itc)} of a capture investment, which changes "
         "the payback math."),
        tech_line,
        "Would you have 20 minutes next week for us to walk through "
        "the numbers.",
    ])

    return {
        "facility_name": name,
        "owner": owner,
        "province": str(facility.get("province", "")),
        "sector": str(facility.get("sector", "")),
        "emissions_label": fmt_mt(emissions),
        "liability_label": fmt_money_cad(liability),
        "itc_label": fmt_money_cad(itc),
        "readiness": readiness,
        "hooks": hooks,
        "talking_points": talking_points,
        "email_subject": email_subject,
        "email_body": email_body,
    }


def brief_to_html(brief: dict) -> str:
    """Render the brief as a standalone HTML document for download."""
    r = brief["readiness"]
    score = f"{r['total']:.0f} of {r['denominator']}"
    hooks = "".join(f"<li>{h}</li>" for h in brief["hooks"])
    points = "".join(f"<li>{p}</li>" for p in brief["talking_points"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>Outreach brief, {brief['facility_name']}</title>
<style>
body {{ font-family: Georgia, serif; max-width: 700px; margin: 40px auto;
padding: 0 20px; color: #1a1a1a; }}
.kicker {{ letter-spacing: 3px; font-size: 11px; color: #666; }}
h1 {{ margin: 8px 0 4px 0; }}
.meta {{ color: #555; margin-bottom: 24px; }}
.stats {{ display: flex; gap: 24px; margin: 20px 0; flex-wrap: wrap; }}
.stat .l {{ font-size: 11px; color: #666; }}
.stat .v {{ font-size: 22px; font-weight: bold; }}
h2 {{ border-bottom: 2px solid #1a1a1a; padding-bottom: 4px; }}
li {{ margin: 6px 0; }}
.foot {{ margin-top: 32px; font-size: 12px; color: #777; font-style: italic; }}
</style></head>
<body>
<div class="kicker">OUTREACH BRIEF</div>
<h1>{brief['facility_name']}</h1>
<div class="meta">{brief['owner']}, {brief['province']}, {brief['sector']}</div>
<div class="stats">
<div class="stat"><div class="l">Emissions</div><div class="v">{brief['emissions_label']}</div></div>
<div class="stat"><div class="l">2030 liability</div><div class="v">{brief['liability_label']}</div></div>
<div class="stat"><div class="l">ITC value</div><div class="v">{brief['itc_label']}</div></div>
<div class="stat"><div class="l">Readiness</div><div class="v">{score}</div></div>
</div>
<h2>The hook</h2>
<ul>{hooks}</ul>
<h2>Talking points</h2>
<ul>{points}</ul>
<div class="foot">Screening estimates from public emissions data. Hub distances
are estimates from public approximate locations. Not engineering advice.</div>
</body></html>"""


# ---------------------------------------------------------------------------
# Watchlist and alerts persistence
# ---------------------------------------------------------------------------

def read_watchlist(path: str | None = None) -> set:
    """Read watched facility ids. Missing or corrupt files read as empty."""
    path = path or default_path("watchlist.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            ids = data.get("facility_ids", [])
        elif isinstance(data, list):
            ids = data
        else:
            return set()
        return {str(i) for i in ids}
    except (OSError, ValueError, TypeError):
        return set()


def write_watchlist(ids, path: str | None = None) -> None:
    """Persist watched facility ids with an atomic write."""
    path = path or default_path("watchlist.json")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = json.dumps({"facility_ids": sorted(str(i) for i in ids)},
                         indent=2)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".watchlist_",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_alerts(path: str | None = None) -> list:
    """Read the alert feed. Missing or corrupt files read as empty."""
    path = path or default_path("alerts.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Streamlit views. Streamlit is imported lazily so the pure logic above
# stays importable without a Streamlit runtime.
# ---------------------------------------------------------------------------

def _needs_company(df) -> bool:
    return "company_trade" in df.columns


def render_owner_portfolios(df: pd.DataFrame) -> None:
    import streamlit as st
    st.subheader("Owner Portfolios")
    st.write("One account conversation instead of many plant conversations. "
             "Sell to the owner, not the stack.")
    if not _needs_company(df):
        st.info("Owner portfolios need company data, which this model "
                "version does not include. Switch to v3 staging to use "
                "this tool.")
        return
    owners = owner_portfolios(df)
    if owners.empty:
        st.info("No facilities to aggregate.")
        return
    table = owners[["company", "n_facilities", "total_emissions_t",
                    "total_liability_2030_cad", "total_itc_cad",
                    "tier_unanimous", "tier_majority", "tier_contested"]]
    st.dataframe(
        table.rename(columns={
            "company": "Company",
            "n_facilities": "Facilities",
            "total_emissions_t": "Emissions (t/yr)",
            "total_liability_2030_cad": "2030 liability (CAD)",
            "total_itc_cad": "ITC value (CAD)",
            "tier_unanimous": "Unanimous",
            "tier_majority": "Majority",
            "tier_contested": "Contested",
        }),
        use_container_width=True,
        hide_index=True,
    )
    pick = st.selectbox("Open an account",
                        owners["company"].tolist(),
                        key="bd_owner_pick")
    row = owners[owners["company"] == pick].iloc[0]
    st.subheader(pick)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Facilities", int(row["n_facilities"]))
    c2.metric("Emissions", fmt_mt(row["total_emissions_t"]))
    c3.metric("2030 liability",
              fmt_money_cad(row["total_liability_2030_cad"]))
    c4.metric("ITC value", fmt_money_cad(row["total_itc_cad"]))
    st.write("Why this account")
    for b in why_this_account(row):
        st.write("• " + b)
    facs = df[df["company_trade"].apply(normalize_company) == pick]
    st.dataframe(
        facs[["facility_name", "city", "sector", "avg_annual_emissions",
              "verdict_tier"]].rename(columns={
                  "facility_name": "Facility",
                  "city": "City",
                  "sector": "Sector",
                  "avg_annual_emissions": "Emissions (t/yr)",
                  "verdict_tier": "Tier",
              }),
        use_container_width=True,
        hide_index=True,
    )


def render_readiness(df: pd.DataFrame, hubs_path: str | None = None,
                     cities_path: str | None = None) -> None:
    import streamlit as st
    st.subheader("Readiness")
    st.write("Can we physically build capture here. Scored in the open, "
             "so the number survives diligence.")
    hubs = load_hubs(hubs_path)
    cities = load_city_coords(cities_path)
    rows = []
    for _, r in df.iterrows():
        s = readiness_score(r["avg_annual_emissions"], r["co2_share"],
                            r.get("city", ""), hubs, cities)
        rows.append({
            "Facility": r["facility_name"],
            "City": r.get("city", ""),
            "Score": f"{s['total']:.0f} of {s['denominator']}",
            "Scale (40)": s["scale_pts"],
            "Purity (30)": s["purity_pts"],
            "Proximity (30)": (s["prox_pts"]
                               if s["prox_pts"] is not None else "unknown"),
            "Nearest hub": s["nearest_hub"] or "unknown",
            "Distance (km)": (s["prox_km"]
                              if s["prox_km"] is not None else "unknown"),
        })
    table = pd.DataFrame(rows)
    order = table["Score"].str.extract(r"([\d.]+)").astype(float)
    table = table.iloc[order[0].sort_values(ascending=False).index]
    st.dataframe(table, use_container_width=True, hide_index=True)
    with st.expander("How the score works"):
        st.write("Scale is worth 40 points. Four points per megatonne of "
                 "average annual emissions, capped at 40.")
        st.write("Purity is worth 30 points. The CO2 share times 30.")
        st.write("Proximity is worth 30 points. Under 80 km to a hub "
                 "scores 30, under 150 scores 20, under 300 scores 10, "
                 "otherwise 0.")
        st.write("Hub locations are public approximate points, so "
                 "distances are estimates. Cities we cannot place score "
                 "out of 70, with a note on the row.")
        st.write("Hub sources:")
        for h in hubs:
            st.write("• " + h["name"] + ". " + h["source"] + ".")


def render_scenario_lab(df: pd.DataFrame) -> None:
    import streamlit as st
    st.subheader("Scenario Lab")
    st.write("Move the assumptions. Watch the economics move.")
    price = st.slider("Carbon price in 2030, dollars per tonne",
                      50, 200, 115, 5, key="bd_price")
    itc_rate = st.slider("ITC rate, percent of eligible investment",
                         0, 60, 50, 1, key="bd_itc")
    mult = st.slider("Capture cost multiplier, 1.0 is the base case",
                     0.7, 1.5, 1.0, 0.05, key="bd_mult")
    rows = []
    for _, r in df.iterrows():
        s = scenario_row(r["avg_annual_emissions"], r["sector"],
                         r["co2_share"], price, itc_rate, mult)
        rows.append({
            "Facility": r["facility_name"],
            "Tier": r.get("verdict_tier", ""),
            "2030 liability": s["liability_cad"],
            "ITC value": s["itc_credit_cad"],
            "Margin": s["margin_cad"],
            "Verdict": s["verdict"],
        })
    t = pd.DataFrame(rows).sort_values("Margin", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio 2030 liability",
              fmt_money_cad(t["2030 liability"].sum()))
    c2.metric("Portfolio ITC value", fmt_money_cad(t["ITC value"].sum()))
    c3.metric("Economic facilities",
              int((t["Verdict"] == "Economic").sum()))
    st.dataframe(t, use_container_width=True, hide_index=True)
    st.caption("Capture cost is implied as abatable value minus base "
               "margin from the economics module, then scaled. ITC at "
               "the custom rate keeps the modeled capex split and rate "
               "structure.")


def render_outreach_kit(df: pd.DataFrame, hubs_path: str | None = None,
                        cities_path: str | None = None) -> None:
    import streamlit as st
    st.subheader("Outreach Kit")
    st.write("One page per facility that a rep can send. The numbers do "
             "the opening.")
    labels = [f"{r['facility_name']} ({r['facility_id']})"
              for _, r in df.iterrows()]
    pick = st.selectbox("Facility", labels, key="bd_outreach_pick")
    fid = pick.rsplit("(", 1)[1].rstrip(")")
    row = df[df["facility_id"].astype(str) == str(fid)].iloc[0]
    hubs = load_hubs(hubs_path)
    cities = load_city_coords(cities_path)
    readiness = readiness_score(row["avg_annual_emissions"],
                               row["co2_share"], row.get("city", ""),
                               hubs, cities)
    brief = outreach_brief(row.to_dict(), readiness)
    st.subheader(brief["facility_name"])
    st.write(f"{brief['owner']}, {brief['province']}, {brief['sector']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Emissions", brief["emissions_label"])
    c2.metric("2030 liability", brief["liability_label"])
    c3.metric("ITC value", brief["itc_label"])
    r = brief["readiness"]
    c4.metric("Readiness", f"{r['total']:.0f} of {r['denominator']}")
    st.write("The hook")
    for h in brief["hooks"]:
        st.write("• " + h)
    st.write("Talking points")
    for i, p in enumerate(brief["talking_points"], 1):
        st.write(f"{i}. {p}")
    with st.expander("Outreach email, ready to copy"):
        st.text(brief["email_subject"] + "\n\n" + brief["email_body"])
    st.download_button("Download brief as HTML",
                       brief_to_html(brief),
                       file_name=f"outreach_{fid}.html",
                       mime="text/html",
                       key="bd_outreach_dl")


def render_watchlist(df: pd.DataFrame, watchlist_path: str | None = None,
                     alerts_path: str | None = None) -> None:
    import streamlit as st
    st.subheader("Watchlist")
    st.write("Star the targets that matter. The checker script watches "
             "the data and writes here.")
    watched = read_watchlist(watchlist_path)
    labels = [f"{r['facility_name']} ({r['facility_id']})"
              for _, r in df.iterrows()]
    pick = st.selectbox("Facility", labels, key="bd_watch_pick")
    fid = pick.rsplit("(", 1)[1].rstrip(")")
    starred = fid in watched
    if st.button("Remove from watchlist" if starred else "Add to watchlist",
                 key="bd_watch_toggle"):
        if starred:
            watched.discard(fid)
        else:
            watched.add(fid)
        write_watchlist(watched, watchlist_path)
        st.rerun()
    if watched:
        w = df[df["facility_id"].isin(watched)][
            ["facility_id", "facility_name", "company_trade",
             "avg_annual_emissions"]]
        st.dataframe(w.rename(columns={
            "facility_id": "ID", "facility_name": "Facility",
            "company_trade": "Company",
            "avg_annual_emissions": "Emissions (t/yr)"}),
            use_container_width=True, hide_index=True)
    else:
        st.info("Nothing watched yet. Pick a facility above and add it.")
    st.subheader("Alerts")
    alerts = read_alerts(alerts_path)
    if not alerts:
        st.info("No alerts yet. The checker script appends here when the "
                "data vintage changes.")
    else:
        for a in reversed(alerts[-20:]):
            st.write("• [" + str(a.get("severity", "")) + "] "
                     + str(a.get("text", "")))
    st.caption("Alert email delivery is not built yet. That is a follow up.")
