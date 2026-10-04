"""Capstone deployment app: CCS/CU priority screening for Alberta facilities."""
import hashlib
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

GHGRP_URL = "https://open.canada.ca/data/en/dataset/a8ba14b7-7f23-462a-bdbb-83b0ef629823"
TIER_URL = "https://www.alberta.ca/technology-innovation-and-emissions-reduction-regulation"

st.set_page_config(page_title="CCS Priority Screening", layout="wide")

bundle = joblib.load("model/model.pkl")
tree, le = bundle["model"], bundle["encoder"]
FEATURES = bundle["features"]
fac = pd.read_csv("model/facilities.csv")
yearly = pd.read_csv("data/Capstone_Dataset_clean.csv")

# Readable labels: a few facilities are filed under a bare ID code,
# so show the operator name with the code in brackets.
display = {}
for _, r in fac.iterrows():
    nm = str(r["facility_name"])
    if re.match(r"^[A-Z0-9]{6,}$", nm):
        y = yearly[yearly["facility_id"] == r["facility_id"]].sort_values("year")
        tn = y.iloc[-1]["company_trade"] if len(y) and pd.notna(y.iloc[-1]["company_trade"]) else nm
        display[r["facility_id"]] = f"{tn} ({nm})"
    else:
        display[r["facility_id"]] = nm

h1, h2 = st.columns([6, 1])
with h1:
    st.title("Carbon Capture Priority Screening")
with h2:
    st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
    st.toggle("Dark mode", key="dark_mode")

dark = st.session_state.get("dark_mode", False)
INK = "#f5f5f7" if dark else "#1d1d1f"
GRAY = "#a1a1a6" if dark else "#6e6e73"
TRACK = "#3a3a3c" if dark else "#e8e8ed"
RED = "#ff453a" if dark else "#e02020"
AMBER = "#ffd60a" if dark else "#e8930c"
ACCENT = "#0a84ff" if dark else "#0071e3"
PAGE_BG = "#000000" if dark else "#ffffff"
PANEL_BG = "#1c1c1e" if dark else "#f5f5f7"

if dark:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #000000; }
    [data-testid="stHeader"] { background-color: transparent; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
    h1, h2, h3, [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
        color: #f5f5f7 !important;
    }
    hr { border-color: #3a3a3c !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #1c1c1e !important;
        border-color: #3a3a3c !important;
    }
    div[data-baseweb="select"] span { color: #f5f5f7 !important; }
    div[data-baseweb="tag"] { background-color: #2c2c2e !important; }
    div[data-baseweb="tag"] span { color: #f5f5f7 !important; }
    button[data-baseweb="tab"] p { color: #a1a1a6 !important; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #ff453a !important; }
    [data-testid="stExpander"] { background-color: #1c1c1e !important;
        border-color: #3a3a3c !important; }
    [data-testid="stExpander"] summary p { color: #f5f5f7 !important; }
    [data-testid="stDataFrame"] { background-color: #1c1c1e; }
    section[data-testid="stSlider"] p { color: #f5f5f7 !important; }
    </style>
    """, unsafe_allow_html=True)


def explain_path(row):
    """Walk the tree for one facility, return plain-word steps."""
    t = tree.tree_  # the underlying tree struct holds children/feature/threshold
    x = row[FEATURES].values.reshape(1, -1)
    node = 0
    steps = []
    while t.children_left[node] != -1:
        f = FEATURES[t.feature[node]]
        thr = t.threshold[node]
        val = float(x[0, t.feature[node]])
        go_left = val <= thr
        if f == "log_emissions":
            tonnes = np.expm1(thr)
            steps.append(
                f"Average emissions {'below' if go_left else 'above'} "
                f"{tonnes:,.0f} tonnes per year.")
        elif f == "naics_sector_encoded":
            side = le.classes_[le.transform(le.classes_) <= thr] if go_left else \
                le.classes_[le.transform(le.classes_) > thr]
            shown = ", ".join(side[:4])
            more = f" and {len(side) - 4} more" if len(side) > 4 else ""
            steps.append(
                f"Sector is in the group: {shown}{more}."
                if go_left else
                f"Sector is not in the lower group, it is one of: {shown}{more}.")
        else:
            steps.append(
                f"Reported for {'at most' if go_left else 'more than'} "
                f"{thr:.0f} years.")
        node = t.children_left[node] if go_left else t.children_right[node]
    return steps


def priority_badge(priority):
    color = RED if priority == "CCS Candidate" else AMBER
    st.markdown(
        f"<div style='font-size:42px;font-weight:700;color:{color};"
        f"letter-spacing:-0.5px;line-height:1.1;'>{priority}</div>",
        unsafe_allow_html=True)
    if priority == "CCS Candidate":
        st.markdown(
            f"<div style='color:{GRAY};font-size:16px;margin-top:6px;'>"
            f"High-concentration CO2 stream. Built for capture and storage [3].</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='color:{GRAY};font-size:16px;margin-top:6px;'>"
            f"Mixed gas stream. Better directed toward carbon utilization [3].</div>",
            unsafe_allow_html=True)


def share_bar(share):
    """CO2 share against the 85 percent CCS cutoff. The decision, drawn."""
    pct = share * 100
    color = RED if share >= 0.85 else AMBER
    st.markdown(
        f"<div style='margin:10px 0 2px 0;'>"
        f"<div style='position:relative;height:10px;background:{TRACK};border-radius:5px;'>"
        f"<div style='position:absolute;left:0;top:0;height:10px;width:{pct:.1f}%;"
        f"background:{color};border-radius:5px;'></div>"
        f"<div style='position:absolute;left:85%;top:-4px;width:2px;height:18px;"
        f"background:{ACCENT};'></div>"
        f"</div>"
        f"<div style='display:flex;justify-content:space-between;font-size:13px;"
        f"color:{GRAY};margin-top:8px;'>"
        f"<span>CO2 share: <b style='color:{INK};'>{pct:.0f}%</b></span>"
        f"<span style='color:{ACCENT};'>85% CCS cutoff</span>"
        f"</div></div>",
        unsafe_allow_html=True)


def peer_bar(value, median, sector):
    """This facility against its sector median. Context for the decision."""
    mx = max(value, median, 1)
    st.markdown(
        f"<div style='margin:14px 0 2px 0;font-size:13px;color:{GRAY};'>"
        f"Average annual emissions vs {sector} median</div>"
        f"<div style='margin:6px 0;'>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
        f"<div style='width:110px;font-size:13px;color:{GRAY};'>This facility</div>"
        f"<div style='flex:1;height:10px;background:{TRACK};border-radius:5px;'>"
        f"<div style='height:10px;width:{100*value/mx:.1f}%;background:{INK};"
        f"border-radius:5px;'></div></div>"
        f"<div style='width:110px;font-size:13px;color:{INK};text-align:right;'>"
        f"{value:,.0f} t</div></div>"
        f"<div style='display:flex;align-items:center;gap:10px;'>"
        f"<div style='width:110px;font-size:13px;color:{GRAY};'>Sector median</div>"
        f"<div style='flex:1;height:10px;background:{TRACK};border-radius:5px;'>"
        f"<div style='height:10px;width:{100*median/mx:.1f}%;background:{GRAY};"
        f"border-radius:5px;'></div></div>"
        f"<div style='width:110px;font-size:13px;color:{GRAY};text-align:right;'>"
        f"{median:,.0f} t</div></div>"
        f"</div>",
        unsafe_allow_html=True)


def facility_profile(row):
    """Full profile for one facility row."""
    # Company and city live in the yearly file, not the facility rollup.
    y = yearly[yearly["facility_id"] == row["facility_id"]].sort_values("year")
    latest = y.iloc[-1] if len(y) else None

    def _val(col):
        if latest is not None and col in y.columns and pd.notna(latest[col]):
            return latest[col]
        return "Not reported"

    st.subheader(display[row["facility_id"]])
    a, b = st.columns(2)
    with a:
        st.write(f"**Sector:** {row['sector']}")
        st.write(f"**Company:** {_val('company_legal')}")
        st.write(f"**City:** {_val('city')}")
        st.write(f"**NAICS code:** {int(row['naics_code'])}")
    with b:
        st.write(f"**Average annual emissions:** {row['avg_annual_emissions']:,.0f} tonnes CO2e")
        st.write(f"**Average CO2:** {row['avg_co2']:,.0f} tonnes")
        st.write(f"**CO2 share:** {row['co2_share']:.1%}, "
                 f"CH4 share: {row['ch4_share']:.1%}, N2O share: {row['n2o_share']:.1%}")
        st.write(f"**Years reported:** {int(row['years_reported'])}")
        st.write(f"**TIER band:** {row['emission_band']}")
    st.write("**Emissions history**")
    hist = yearly[yearly["facility_id"] == row["facility_id"]].sort_values("year")
    st.line_chart(hist.set_index("year")["total_emissions"])
    x = row[FEATURES].values.reshape(1, -1)
    st.write("**Screening result**")
    priority_badge(tree.predict(x)[0])
    st.write("**Why this result**")
    for i, s in enumerate(explain_path(row), 1):
        st.write(f"{i}. {s}")


def references():
    st.divider()
    st.subheader("References")
    st.write(f"[1] Environment and Climate Change Canada. Greenhouse Gas "
             f"Reporting Program, facility emissions data 2004 to present, "
             f"dataset PDGES-GHGRP-GHGEmissionsGES-2004-Present. {GHGRP_URL}")
    st.write(f"[2] Government of Alberta. Technology Innovation and Emissions "
             f"Reduction (TIER) Regulation. Facilities emitting 100,000 tonnes "
             f"CO2e or more per year are regulated. {TIER_URL}")
    st.write("[3] Alberta Innovates guidance that high-concentration CO2 streams "
             "are suited to carbon capture and storage while mixed-gas streams "
             "suit carbon utilization, as referenced in the AMII AI Pathways "
             "Technical Track capstone materials.")
    st.write("[4] AMII. AI Pathways Technical Track, Capstone Project: "
             "Prioritizing Industrial Sectors and Facilities for Carbon Capture "
             "and Carbon Utilization Deployment in Alberta Using Emissions Data. 2026.")


tab_about, tab_screen, tab_facilities = st.tabs(["About", "Screening", "Facilities"])

with tab_about:
    st.subheader("What this is")
    st.write("Alberta releases about 270 megatonnes of CO2 every year. Carbon "
             "capture can trap it before it reaches the air, but it is expensive "
             "and cannot go everywhere. The real question is simple. Where should "
             "it go first.")
    st.write("This tool answers that as a first pass. It reads public emissions "
             "data for 1,199 Alberta industrial facilities and screens the best "
             "candidates for carbon capture or carbon utilization. What used to "
             "take months of manual review now takes minutes, across every "
             "facility, not just the big names.")
    st.subheader("How it works")
    st.write("1. Start with public data. Every figure comes from Environment and "
             "Climate Change Canada's Greenhouse Gas Reporting Program [1].")
    st.write("2. Keep the serious emitters. Facilities averaging at least 100,000 "
             "tonnes of CO2e a year fall under Alberta's TIER regulation [2]. "
             "That leaves 150.")
    st.write("3. Sort by stream. Streams that are at least 85% CO2 screen as "
             "carbon capture candidates. Mixed streams screen as carbon "
             "utilization candidates [3].")
    st.write("4. Learn the pattern. A Decision Tree model learns the screening "
             "rules from the data, so any facility can be screened the same "
             "way [4].")
    st.subheader("What this is not")
    st.write("This is a triage screen, not an engineering verdict. It does not "
             "replace site studies, cost analysis, or geology. It tells you where "
             "the expensive reviews should start.")
    st.subheader("How to use it")
    st.write("On the Screening tab, pick a sector, pick a facility, and see its "
             "result with the reasons in plain words.")
    st.write("On the Facilities tab, browse the full table. Filter it, click a "
             "row for the full profile, and follow the source link to verify "
             "the numbers yourself.")
    references()

with tab_screen:
    st.write("Pick an Alberta industrial facility to see whether it screens as a "
             "carbon capture candidate or a carbon utilization candidate, and why. "
             "Built on public emissions data [1].")
    s1, s2 = st.columns(2)
    with s1:
        sel_sector = st.selectbox("Sector", ["All sectors"] + sorted(fac["sector"].unique()),
                                  key="screen_sector")
    sub = fac if sel_sector == "All sectors" else fac[fac["sector"] == sel_sector]
    ids = sorted(sub["facility_id"].unique(), key=lambda i: display[i].lower())
    with s2:
        choice = st.selectbox("Facility", ids, key="screen_pick",
                              format_func=lambda i: display[i])
    st.caption(f"{len(ids)} facilities")
    row = fac[fac["facility_id"] == choice].iloc[0]
    x = row[FEATURES].values.reshape(1, -1)

    priority_badge(tree.predict(x)[0])
    st.divider()
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader(display[choice])
        st.write(f"**Sector:** {row['sector']}")
        st.write(f"**Average annual emissions:** {row['avg_annual_emissions']:,.0f} tonnes CO2e")
        st.write(f"**CO2 share of emissions:** {row['co2_share']:.0%}")
        st.write(f"**Years reported:** {int(row['years_reported'])}")
        st.write(f"**TIER band:** {row['emission_band']}")
    with right:
        st.subheader("At a glance")
        share_bar(row["co2_share"])
        med = fac[fac["sector"] == row["sector"]]["avg_annual_emissions"].median()
        peer_bar(row["avg_annual_emissions"], med, row["sector"])
    st.subheader("Why this result")
    for i, s in enumerate(explain_path(row), 1):
        st.write(f"{i}. {s}")
    with st.expander("About this model"):
        st.write("Decision Tree classifier, maximum depth 3, trained on 150 "
                 "above-threshold Alberta facilities [2]. Features: log-scaled average "
                 "emissions, encoded industry sector, years reported. Gas shares were "
                 "excluded from features to avoid label leakage. Evaluated with "
                 "5-fold stratified cross validation: accuracy 0.75, macro F1 0.62.")
        st.write("Data: Environment and Climate Change Canada, Greenhouse Gas "
                 "Reporting Program, public dataset 2004 to 2023 [1].")

with tab_facilities:
    st.write("Every screened facility, with its data source and result. All figures "
             "come from Environment and Climate Change Canada's Greenhouse Gas "
             "Reporting Program [1]. Facilities averaging at least 100,000 tonnes "
             "CO2e per year fall under Alberta's TIER regulation [2] and form the "
             "screening population. Select a row for the full facility profile.")
    f1, f2, f3 = st.columns(3)
    with f1:
        q = st.text_input("Search by name")
    with f2:
        sectors = st.multiselect("Sector", sorted(fac["sector"].unique()),
                                 default=sorted(fac["sector"].unique()))
    with f3:
        prios = st.multiselect("Result", ["CCS Candidate", "Potential CU Candidate"],
                               default=["CCS Candidate", "Potential CU Candidate"])
    min_e = st.slider("Minimum average annual emissions (tonnes CO2e)",
                      100_000, int(fac["avg_annual_emissions"].max()),
                      100_000, step=10_000)

    t = fac[(fac["facility_name"].str.contains(q, case=False, na=False, regex=False))
            & (fac["sector"].isin(sectors))
            & (fac["priority"].isin(prios))
            & (fac["avg_annual_emissions"] >= min_e)].copy()
    t["source"] = GHGRP_URL
    t = t.sort_values("avg_annual_emissions", ascending=False).reset_index(drop=True)
    # Key the table to the filter state so a stale row selection is cleared
    # whenever the filters change, instead of pointing at the wrong row.
    sig_src = f"{q}|{'/'.join(sorted(sectors))}|{'/'.join(sorted(prios))}|{min_e}"
    sig = hashlib.md5(sig_src.encode()).hexdigest()[:10]
    st.write(f"{len(t)} facilities")
    event = st.dataframe(
        t[["facility_name", "sector", "avg_annual_emissions", "co2_share",
           "priority", "source"]],
        column_config={
            "facility_name": st.column_config.TextColumn("Facility"),
            "sector": st.column_config.TextColumn("Sector"),
            "avg_annual_emissions": st.column_config.NumberColumn(
                "Avg emissions (tCO2e/yr)", format="%.0f"),
            "co2_share": st.column_config.NumberColumn("CO2 share", format="%.0%%"),
            "priority": st.column_config.TextColumn("Screening result"),
            "source": st.column_config.LinkColumn("Data source",
                                                  display_text="ECCC GHGRP"),
        },
        key=f"fac_table_{sig}",
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        use_container_width=True,
    )
    rows = [r for r in event.selection.rows if r < len(t)]
    if rows:
        sel = t.iloc[rows[0]]
        st.divider()
        facility_profile(fac[fac["facility_id"] == sel["facility_id"]].iloc[0])
