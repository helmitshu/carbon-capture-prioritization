"""Capstone deployment app: CCS/CU priority screening for Canadian facilities."""
import hashlib
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from features import (CO2_CUTOFF, FEATURES, encoder_categories,
                      encoder_kind, encoder_values)
from memo import render_memo_html, render_memo_pdf
from model_client import get_tree_and_encoder, predict_batch

GHGRP_URL = "https://open.canada.ca/data/en/dataset/a8ba14b7-7f23-462a-bdbb-83b0ef629823"
TIER_URL = "https://www.alberta.ca/technology-innovation-and-emissions-reduction-regulation"

st.set_page_config(page_title="CCS Priority Screening", layout="wide")

# Model version switcher. The default comes from DEFAULT_MODEL_VERSION
# ("v2" unless set to "v3"): the production service keeps the v2 default
# while a staging service can set DEFAULT_MODEL_VERSION=v3 to open on the
# national preview. The user can still switch manually afterwards.
_default_label = ("v3 staging"
                  if os.environ.get("DEFAULT_MODEL_VERSION", "v2").strip().lower() == "v3"
                  else "v2 production")
version_choice = st.sidebar.selectbox(
    "Model version",
    ["v2 production", "v3 staging"],
    index=["v2 production", "v3 staging"].index(_default_label),
    key="model_version",
    help="v2 is the frozen production model for Alberta. v3 staging is the "
         "national preview and always runs locally, never through the API.")
IS_V3 = version_choice == "v3 staging"
# Stale facility-scoped state from the other version would crash the
# pickers (its ids are not in this version's table), so drop it on switch.
_prev_version = st.session_state.get("_loaded_version")
if _prev_version is not None and _prev_version != version_choice:
    for _k in ("walk_pick", "walk", "walk_autorun", "screen_sector",
               "screen_pick"):
        st.session_state.pop(_k, None)
st.session_state["_loaded_version"] = version_choice

@st.cache_resource
def _load_v3_bundle(path="model/model_v3.pkl"):
    """Load the v3 model bundle once per process.

    Cached: the bundle is a fixed build artifact. If the file is replaced
    on disk, restart the app (or clear the Streamlit cache) to pick it up.
    """
    _bundle3 = joblib.load(path)
    assert list(_bundle3["features"]) == FEATURES, "v3 feature schema drift"
    return _bundle3["model"], _bundle3["encoder"]


@st.cache_resource(ttl=300)
def _get_tree_and_encoder():
    """Resolve the v2 model once per process (API with local fallback).

    The API-or-local decision is re-evaluated every 5 minutes instead of on
    every rerun, so widget interactions do not pay for a health-check round
    trip. Trade-off: if the API dies mid-session, the app keeps using the
    cached source until the TTL expires or the session reloads.
    """
    return get_tree_and_encoder()


@st.cache_data
def _read_csv(path):
    """Read a data CSV once per process (keyed on path)."""
    return pd.read_csv(path)


if IS_V3:
    # Staging preview only: always the local bundle, never the API.
    tree, le = _load_v3_bundle()
    MODEL_SOURCE = "local (v3 staging)"
    fac = _read_csv("model/facilities_v3.csv")
    yearly = _read_csv("data/Capstone_Dataset_clean_national.csv")
    META = {"n_facilities": len(fac),
            "n_facilities_total": int(yearly["facility_id"].nunique()),
            "n_yearly_rows": len(yearly)}
else:
    # v2 production: served by the prediction API when reachable, with an
    # in-process fallback, so the app never goes down because the API does.
    tree, le, MODEL_SOURCE = _get_tree_and_encoder()
    fac = _read_csv("model/facilities.csv")
    yearly = _read_csv("data/Capstone_Dataset_clean.csv")
    with open("model/meta.json", encoding="utf-8") as _mf:
        META = json.load(_mf)


@st.cache_data
def _v3_data_facts():
    """Facts for the About copy, derived from the loaded data.

    Hardcoded record counts rot on every data refresh, so the About
    tab reads them here instead. v3 only; the v2 line stays frozen.
    """
    facts = {
        "year_min": int(yearly["year"].min()),
        "year_max": int(yearly["year"].max()),
        "clean_rows": len(yearly),
        "clean_facilities": int(yearly["facility_id"].nunique()),
        "n_above_threshold": int(META.get("n_facilities", 0)),
    }
    facts["n_years"] = facts["year_max"] - facts["year_min"] + 1
    try:
        with open("hidden_files/cleaning_report_national.json",
                  encoding="utf-8") as _rf:
            _rep = json.load(_rf)
        facts["raw_rows"] = int(_rep.get("raw_shape", [0])[0])
        facts["raw_facilities"] = int(_rep.get("raw_n_facilities", 0))
        _miss = _rep.get("raw_missing_by_column", {})
        _npri_key = next((k for k in _miss if "NPRI ID" in k), None)
        facts["npri_missing"] = int(_miss[_npri_key]) if _npri_key else 0
    except (OSError, ValueError, KeyError):
        facts["raw_rows"] = facts["clean_rows"]
        facts["raw_facilities"] = facts["clean_facilities"]
        facts["npri_missing"] = 0
    return facts


# Model prediction for every facility, computed once. The Facilities table
# shows this next to the rule-based label so the two never get mixed up.
# One call (not predict+apply) so the remote path costs a single round trip.
# Cached per (model version, model source): the tree and feature frame are
# excluded from the cache key (underscore prefix) because Streamlit cannot
# hash the remote-tree shim, and both are already fixed per version+source.
@st.cache_data
def _startup_predictions(version, source, _tree, _fac_features):
    return predict_batch(_fac_features, _tree)


fac["model_prediction"], _startup_leaves = _startup_predictions(
    version_choice, MODEL_SOURCE, tree, fac[FEATURES])


@st.cache_data
def _file_bytes(path):
    with open(path, "rb") as f:
        return f.read()

# Per-leaf model reliability, computed once from in-sample fit. Leaves where
# the tree is frequently wrong get their own alert. (Raw leaf probabilities
# are distorted by balanced class weights, so the error rate is the honest
# signal, not the probability.)
# DataFrames (not numpy) go to predict/apply so sklearn never emits the
# feature-names warning; the leaves come free from the startup batch call.
_pred_all = fac["model_prediction"].values
_leaf_all = _startup_leaves
_true_all = fac["priority"].values
LEAF_ERR = {}
for lf in set(_leaf_all):
    m = _leaf_all == lf
    LEAF_ERR[int(lf)] = float((_pred_all[m] != _true_all[m]).mean())
HIGH_ERR_THRESHOLD = 0.25
BORDERLINE_PTS = 0.03  # CO2 share within 3 points of the CCS cutoff

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
    st.markdown(
        "<div style='width:56px;height:4px;border-radius:2px;margin-top:4px;"
        "background:linear-gradient(90deg,#0e7c7b,#0071e3);'></div>",
        unsafe_allow_html=True)
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
PAGE_BG = "#000000" if dark else "#f6f7f9"
PANEL_BG = "#1c1c1e" if dark else "#ffffff"

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
    [data-testid="stDownloadButton"] button[kind="secondary"],
    [data-testid="stButton"] button[kind="secondary"] {
        background-color: #1c1c1e !important;
        border-color: #3a3a3c !important;
    }
    [data-testid="stDownloadButton"] button[kind="secondary"] p,
    [data-testid="stButton"] button[kind="secondary"] p {
        color: #f5f5f7 !important;
    }
    </style>
    """, unsafe_allow_html=True)

if not dark:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #f6f7f9; }
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
            _cats = np.array(encoder_categories(le))
            _vals = encoder_values(le)
            side = _cats[_vals <= thr] if go_left else _cats[_vals > thr]
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


def disagreement_banner(pred, row):
    """Flag when the model prediction disagrees with the rule-based label.

    The label comes straight from the data (CO2 share vs the 85% cutoff).
    The model never sees CO2 share, so small facilities with pure CO2
    streams can fool it. Surface the disagreement instead of hiding it.
    """
    rule = row["priority"]
    if pred == rule:
        return
    st.markdown(
        f"<div style='background:{PANEL_BG};border:1px solid {AMBER};"
        f"border-radius:10px;padding:12px 16px;margin:12px 0;'>"
        f"<div style='font-weight:700;color:{AMBER};font-size:15px;'>"
        f"Model and rule disagree. Needs human review.</div>"
        f"<div style='color:{GRAY};font-size:14px;margin-top:4px;'>"
        f"The data says <b style='color:{INK};'>{rule}</b> "
        f"(CO2 share {row['co2_share']:.0%}, cutoff {CO2_CUTOFF:.0%}), but the model "
        f"predicted <b style='color:{INK};'>{pred}</b>. The tree never sees "
        f"the CO2 share directly, so smaller facilities with pure CO2 "
        f"streams can be misclassified.</div></div>",
        unsafe_allow_html=True)


def _alert_box(title, body):
    st.markdown(
        f"<div style='background:{PANEL_BG};border:1px solid {AMBER};"
        f"border-radius:10px;padding:12px 16px;margin:12px 0;'>"
        f"<div style='font-weight:700;color:{AMBER};font-size:15px;'>"
        f"{title}</div>"
        f"<div style='color:{GRAY};font-size:14px;margin-top:4px;'>"
        f"{body}</div></div>",
        unsafe_allow_html=True)


def borderline_banner(row):
    """Flag facilities whose CO2 share sits near the CCS cutoff."""
    gap = abs(row["co2_share"] - CO2_CUTOFF)
    if gap > BORDERLINE_PTS:
        return
    _alert_box(
        "Borderline call. Needs human review.",
        f"CO2 share is {row['co2_share']:.0%}, within "
        f"{BORDERLINE_PTS:.0%} of the {CO2_CUTOFF:.0%} cutoff. The {CO2_CUTOFF:.0%} line is a "
        f"stipulated threshold, not a physical boundary: a small data "
        f"revision would flip this verdict.")


def confidence_banner(pred, row):
    """Flag predictions from tree leaves where the model is often wrong."""
    lf = int(tree.apply(row[FEATURES].to_frame().T)[0])
    err = LEAF_ERR.get(lf, 0.0)
    if err <= HIGH_ERR_THRESHOLD:
        return
    _alert_box(
        "Low model confidence. Needs human review.",
        f"The model predicted <b style='color:{INK};'>{pred}</b>, but "
        f"facilities landing in this part of the tree were misclassified "
        f"{err:.0%} of the time in training. Treat the prediction with "
        f"extra skepticism and lean on the data.")


def tier_badge(row):
    """Panel agreement badge (v3 only: hidden when the column is absent).

    Unanimous green, majority amber, contested red, plus the panel vote
    count. When the panel majority disagrees with the tree verdict, say
    so in one line: that is the review trigger, not a hidden detail.
    """
    if "verdict_tier" not in row.index:
        return
    tier = str(row["verdict_tier"])
    try:
        votes_txt = f"Panel: {int(row['panel_ccs_votes'])}/4"
    except (TypeError, ValueError, KeyError):
        votes_txt = "Panel: n/a"
    green = "#2dd4bf" if dark else "#0f766e"
    color = {"unanimous": green, "majority": AMBER, "contested": RED}.get(
        tier, GRAY)
    st.markdown(
        f"<div style='display:inline-block;background:{PANEL_BG};"
        f"border:1px solid {color};border-radius:8px;padding:6px 12px;"
        f"margin:8px 0;'>"
        f"<span style='font-weight:700;color:{color};font-size:14px;"
        f"text-transform:capitalize;'>{tier}</span>"
        f"<span style='color:{GRAY};font-size:13px;margin-left:8px;'>"
        f"{votes_txt}</span></div>",
        unsafe_allow_html=True)
    if ("panel_majority" in row.index
            and str(row["panel_majority"]) != str(row["model_prediction"])):
        st.caption(f"Panel majority says {row['panel_majority']} but the "
                   f"tree says {row['model_prediction']}. Needs human review.")


def _econ_summary(row):
    """Compact v3-only economics strip on the Screening detail view.

    Sourced public benchmarks with labeled assumptions. The full
    envelope lives in the generated memo.
    """
    from economics import facility_economics
    from assumptions import CARBON_PRICE_DECK, assumption_value
    co2 = (float(row["co2_share"]) if "co2_share" in row.index
           and pd.notna(row["co2_share"]) else 1.0)
    econ = facility_economics(float(row["avg_annual_emissions"]),
                             str(row["sector"]), co2_share=co2)
    band = econ["band"]
    st.divider()
    st.write("**Indicative economics**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**Capture cost band:** "
                 f"${band['low_cad']:,.0f} to ${band['high_cad']:,.0f} per tonne")
        st.caption("Sector public benchmarks, not site engineering.")
    with c2:
        st.write(f"**2030 carbon liability:** "
                 f"~${econ['liability_2030_cad'] / 1e6:,.0f}M per year")
        st.caption(f"At the ${CARBON_PRICE_DECK['schedule'][2030]:,.0f} per tonne federal schedule. "
                   f"~${econ['abatable_liability_2030_cad'] / 1e6:,.0f}M "
                   f"abatable at {econ['co2_share']:.0%} CO2.")
    with c3:
        st.write(f"**Indicative CCUS tax credit:** "
                 f"~${econ['itc']['credit_cad'] / 1e6:,.0f}M refundable")
        _itc_rates = assumption_value("ccus_itc")
        st.caption(f"{_itc_rates['capture_other'] * 100:g}% capture equipment, "
                   f"{_itc_rates['transport_storage_use'] * 100:g}% transport storage and use.")
    st.caption("Illustrative build assumptions. The memo carries the full "
               "envelope with every assumption labeled.")


def _econ_charts(row):
    """v3-only economics visuals on the Screening detail view.

    Three charts, one story: how much of the emissions is CO2, how
    much of the carbon bill capture can address, and whether capture
    beats the carbon price. Figures come from econ_charts builders,
    which read the same economics module as the text.
    """
    from econ_charts import (cost_band_chart, emissions_mix_donut,
                            liability_split_bar)
    from economics import econ_chart_data
    co2 = (float(row["co2_share"]) if "co2_share" in row.index
           and pd.notna(row["co2_share"]) else 1.0)
    args = (float(row["avg_annual_emissions"]), str(row["sector"]), co2)
    liab_total = econ_chart_data(*args)["liability"]["total"]

    fig = emissions_mix_donut(*args)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Non-CO2 gases are shown in CO2-equivalent tonnes, the "
               "same basis the federal carbon price uses. Only the CO2 "
               "slice can be captured.")

    liab_total = econ_chart_data(*args)["liability"]["total"]
    fig = liability_split_bar(*args)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"The full ${liab_total / 1e6:,.1f}M is the real bill: the "
               f"federal price applies per tonne of CO2e. Capture can "
               f"only address the teal slice.")

    fig = cost_band_chart(*args)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Capture beats the tax only below the dashed line, and "
               "only on the CO2 fraction of emissions.")


def _assumptions_panel():
    """Visible assumptions and sources for the v3 economics layer.

    Every figure in the economics strip rests on one of these
    assumptions. Each row shows the value, its source, and when it was
    last verified. Anything past its review period is flagged in plain
    language so it never goes stale quietly.
    """
    from assumptions import assumption_status
    st.divider()
    with st.expander("Assumptions and sources"):
        st.write("Every economics figure above rests on one of these "
                 "assumptions. Flagged rows are past their review date.")
        for a in assumption_status():
            line = f"**{a['name']}:** {a['value_text']} {a['unit']}"
            if a["overdue"]:
                st.warning(f"Due for review. {line}. {a['status_text']}")
            else:
                st.write(line)
            st.caption(f"Source: {a['source']} Last verified "
                       f"{a['last_verified']}. Review {a['review']}.")


def _memo_section(row, val_fn, reasons):
    """One-click screening memo for a facility row. Pure memo.py underneath."""
    st.write("**Screening memo**")
    op = val_fn("company_trade")
    if op == "Not reported":
        op = val_fn("company_legal")
    facility = {
        "facility_name": str(row["facility_name"]),
        "city": str(val_fn("city")),
        "province": (str(row["province"])
                     if "province" in row.index and pd.notna(row["province"])
                     else "Alberta"),
        "operator": str(op),
        "sector": str(row["sector"]),
        "avg_annual_emissions": float(row["avg_annual_emissions"]),
        "co2_share": float(row["co2_share"]),
        "years_reported": int(row["years_reported"]),
        "priority": str(row["priority"]),
        "model_prediction": str(row["model_prediction"]),
        "data_through": str(int(yearly["year"].max())),
    }
    tier = (str(row["verdict_tier"])
            if "verdict_tier" in row.index and pd.notna(row["verdict_tier"])
            else "")
    votes_raw = row["panel_ccs_votes"] if "panel_ccs_votes" in row.index else 0
    votes = int(votes_raw) if pd.notna(votes_raw) else 0
    mkey = f"memo_html_{row['facility_id']}"
    if st.button("Generate screening memo",
                 key=f"genmemo_{row['facility_id']}"):
        st.session_state[mkey] = render_memo_html(
            facility, reasons, tier, votes,
            economics="v31" if IS_V3 else "legacy")
    if mkey in st.session_state:
        pkey = mkey + "__pdf"
        if pkey not in st.session_state:
            try:
                st.session_state[pkey] = render_memo_pdf(
                    st.session_state[mkey])
            except ImportError:
                st.session_state[pkey] = None
        pdf = st.session_state[pkey]
        if pdf is not None:
            st.download_button(
                "Download screening memo (PDF)", pdf,
                file_name=f"screening_memo_{row['facility_id']}.pdf",
                mime="application/pdf",
                key=f"dlpdf_{row['facility_id']}")
        else:
            st.download_button(
                "Download screening memo (HTML)", st.session_state[mkey],
                file_name=f"screening_memo_{row['facility_id']}.html",
                mime="text/html",
                key=f"dlhtml_{row['facility_id']}")
            st.caption("PDF needs WeasyPrint; running locally produces "
                       "the PDF.")


def share_bar(share):
    """CO2 share against the CCS cutoff. The decision, drawn."""
    pct = share * 100
    cutoff_pct = CO2_CUTOFF * 100
    color = RED if share >= CO2_CUTOFF else AMBER
    st.markdown(
        f"<div style='margin:10px 0 2px 0;'>"
        f"<div style='position:relative;height:10px;background:{TRACK};border-radius:5px;'>"
        f"<div style='position:absolute;left:0;top:0;height:10px;width:{pct:.1f}%;"
        f"background:{color};border-radius:5px;'></div>"
        f"<div style='position:absolute;left:{cutoff_pct:.1f}%;top:-4px;width:2px;height:18px;"
        f"background:{ACCENT};'></div>"
        f"</div>"
        f"<div style='display:flex;justify-content:space-between;font-size:13px;"
        f"color:{GRAY};margin-top:8px;'>"
        f"<span>CO2 share: <b style='color:{INK};'>{pct:.0f}%</b></span>"
        f"<span style='color:{ACCENT};'>{CO2_CUTOFF:.0%} CCS cutoff</span>"
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
        st.write(f"**{'Threshold' if IS_V3 else 'TIER'} band:** {row['emission_band']}")
    st.write("**Emissions history**")
    hist = yearly[yearly["facility_id"] == row["facility_id"]].sort_values("year")
    st.line_chart(hist.set_index("year")["total_emissions"])
    st.write("**Screening result**")
    pred = row["model_prediction"]
    priority_badge(pred)
    disagreement_banner(pred, row)
    borderline_banner(row)
    confidence_banner(pred, row)
    tier_badge(row)
    st.write("**Why this result**")
    for i, s in enumerate(explain_path(row), 1):
        st.write(f"{i}. {s}")
    verdict_reasons = [str(row[c]).strip()
                       for c in ("reason_1", "reason_2", "reason_3")
                       if c in row.index and pd.notna(row[c])
                       and str(row[c]).strip()]
    if verdict_reasons:
        st.write("**Why this verdict**")
        for i, r in enumerate(verdict_reasons, 1):
            st.write(f"{i}. {r}")
    _memo_section(row, _val, verdict_reasons)


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


# ---- How-it-works: visual tree walkthrough ----
# Leaf predictions from the data itself (every facility in a leaf gets the
# same prediction by construction).
_LEAF_PRED, _LEAF_N = {}, {}
for _lf in set(_leaf_all):
    _m = _leaf_all == _lf
    _LEAF_PRED[int(_lf)] = _pred_all[_m][0]
    _LEAF_N[int(_lf)] = int(_m.sum())


def tree_walk(row):
    """Structured walk from root to leaf for one facility.

    Returns (steps, leaf) where each step has the node id, the question in
    plain words, the facility's answer, and which branch it took.
    """
    t = tree.tree_
    x = row[FEATURES].values
    node, steps = 0, []
    while t.children_left[node] != -1:
        f = FEATURES[t.feature[node]]
        thr = t.threshold[node]
        val = float(x[t.feature[node]])
        go_left = val <= thr
        if f == "log_emissions":
            q = f"Are average emissions \u2264 {np.expm1(thr):,.0f} tonnes/yr?"
            a = f"{np.expm1(val):,.0f} tonnes/yr"
        elif f == "naics_sector_encoded":
            _cats = np.array(encoder_categories(le))
            _vals = encoder_values(le)
            side = _cats[_vals <= thr] if go_left else _cats[_vals > thr]
            if encoder_kind(le) == "target":
                q = (f"Is the sector candidate rate \u2264 {thr:.0%}? "
                     f"({len(side)} sectors go {'left' if go_left else 'right'})")
                a = f"{row['sector']} (rate {val:.0%})"
            else:
                q = (f"Is the sector code \u2264 {thr:.0f}? "
                     f"({len(side)} sectors go {'left' if go_left else 'right'})")
                a = f"{row['sector']} (code {val:.0f})"
        else:
            q = f"Was it reported for \u2264 {thr:.0f} years?"
            a = f"{val:.0f} years"
        steps.append({"node": node, "feat": f, "thr": thr, "val": val,
                      "q": q, "a": a, "left": go_left})
        node = t.children_left[node] if go_left else t.children_right[node]
    return steps, node


def _band_thresholds():
    """All size-split thresholds in the tree, descending (tonnes/yr)."""
    t = tree.tree_
    return sorted({float(np.expm1(t.threshold[n])) for n in range(t.node_count)
                   if t.children_left[n] != -1
                   and FEATURES[t.feature[n]] == "log_emissions"},
                  reverse=True)


def _size_body(row, group):
    val = float(np.expm1(row["log_emissions"]))
    thrs = _band_thresholds()
    names = ["giant", "large", "medium", "small"]
    band = names[min(sum(1 for th in thrs if val <= th), 3)]
    cuts = ", ".join(f"{np.expm1(s['thr']):,.0f}" for s in group)
    answers = ", ".join("yes" if s["left"] else "no" for s in group)
    name = display[row["facility_id"]]
    n = len(group)
    return (f"The tree's first question is always about size. It tests "
            f"{'it' if n == 1 else f'{n} cuts'} ({cuts}). At {val:,.0f} "
            f"tonnes/yr, {name} answers {answers} and lands in the "
            f"<b>{band}</b> band.")


def _sector_body(row, group, leaf):
    s = group[-1]
    thr, go_left = s["thr"], s["left"]
    _cats = np.array(encoder_categories(le))
    _vals = encoder_values(le)
    side = _cats[_vals <= thr] if go_left else _cats[_vals > thr]
    shown = ", ".join(side[:3])
    more = f" and {len(side) - 3} more" if len(side) > 3 else ""
    pred = _LEAF_PRED[leaf]
    name = display[row["facility_id"]]
    return (f"{name} operates in <b>{row['sector']}</b>. The tree lumps "
            f"sectors into two groups at this split; this one goes with "
            f"{shown}{more}, on the <b>{pred}</b> side.")


def _years_body(row, step):
    yrs = int(row["years_reported"])
    name = display[row["facility_id"]]
    return (f"{name} reported for <b>{yrs} years</b>. In this tree, "
            f"reporting history never decides a split.")


def story_cards(row):
    """Group the walked nodes into human reason cards.

    Consecutive splits on the same feature (e.g. three size cuts) become
    one card telling a single reason, instead of three robot questions.
    Returns (cards, leaf, path_nodes); each card is (title, body_html).
    """
    steps, leaf = tree_walk(row)
    cards = []
    i = 0
    while i < len(steps):
        feat = steps[i]["feat"]
        j = i
        while j < len(steps) and steps[j]["feat"] == feat:
            j += 1
        group = steps[i:j]
        if feat == "log_emissions":
            cards.append(("Size check", _size_body(row, group)))
        elif feat == "naics_sector_encoded":
            cards.append(("Sector check", _sector_body(row, group, leaf)))
        else:
            cards.append(("Reporting history", _years_body(row, group[0])))
        i = j
    path = [s["node"] for s in steps] + [leaf]
    return cards, leaf, path


def reason_card(title, body, i, n):
    st.markdown(
        f"<div class='wstep'><div class='wstep-num'>Reason {i} of {n} · {title}</div>"
        f"<div class='wstep-a'>{body}</div></div>",
        unsafe_allow_html=True)


def _flag_list(row, leaf):
    """Plain-words list of review flags for the decision box."""
    flags = []
    pred = row["model_prediction"]
    if pred != row["priority"]:
        flags.append(
            f"Model vs data: the data says <b>{row['priority']}</b> "
            f"(CO2 share {row['co2_share']:.0%}) but the model predicted "
            f"<b>{pred}</b>.")
    if abs(row["co2_share"] - CO2_CUTOFF) <= BORDERLINE_PTS:
        flags.append(
            f"Borderline: CO2 share is within {BORDERLINE_PTS:.0%} of the "
            f"{CO2_CUTOFF:.0%} cutoff, so a small data revision flips the call.")
    err = LEAF_ERR.get(leaf, 0.0)
    if err > HIGH_ERR_THRESHOLD:
        flags.append(
            f"Shaky ground: facilities in this corner of the tree were "
            f"misclassified {err:.0%} of the time in training.")
    yrs = int(row["years_reported"])
    if yrs <= 5:
        flags.append(
            f"Thin data: only {yrs} years reported, so the average is less stable.")
    return flags


def decision_box(row, leaf):
    """The payoff: a straight recommendation from the flags."""
    pred = row["model_prediction"]
    flags = _flag_list(row, leaf)
    n = len(flags)
    if n == 0:
        title, color = "Decision: clear to proceed", ACCENT
        body = (f"No flags. The data, the model, and the track record agree: "
                f"<b>{pred}</b>. Recommended next step: add it to the {pred} "
                f"review queue. Standard engineering review applies.")
    elif n == 1:
        title, color = "Decision: proceed with caution", AMBER
        body = "One flag needs a human eye before this moves forward."
    else:
        title, color = f"Decision: human review required ({n} flags)", RED
        body = "Do not act on the model output alone. Resolve these flags first."
    bullets = "".join(f"<li>{f}</li>" for f in flags)
    st.markdown(
        f"<div class='dbox' style='border-color:{color};'>"
        f"<div class='dbox-title' style='color:{color};'>{title}</div>"
        f"<div class='dbox-body'>{body}</div>"
        + (f"<ul>{bullets}</ul>" if bullets else "")
        + "</div>",
        unsafe_allow_html=True)


def draw_tree_diagram(path_nodes=()):
    """Render the decision tree; nodes on the walked path are highlighted."""
    t = tree.tree_
    pos, counter = {}, [0]

    def place(n, d):
        l, r = t.children_left[n], t.children_right[n]
        if l == -1:
            x = counter[0]
            counter[0] += 1
        else:
            place(l, d + 1)
            place(r, d + 1)
            x = (pos[l][0] + pos[r][0]) / 2
        pos[n] = (x, -d)

    place(0, 0)
    path = set(path_nodes)
    edge_dim = "#3a4048" if dark else "#c9ccd1"
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PAGE_BG)
    ax.set_facecolor(PAGE_BG)
    for n in range(t.node_count):
        x, y = pos[n]
        on = n in path
        is_leaf = t.children_left[n] == -1
        if is_leaf:
            pred = _LEAF_PRED[n]
            base = RED if pred == "CCS Candidate" else AMBER
            txt = f"{'CCS' if pred == 'CCS Candidate' else 'CU'}\nn={_LEAF_N[n]}"
            if dark:
                fc, alpha = base, 0.30 if on else 0.12
            else:
                fc, alpha = base, 0.16 if on else 0.08
            ec, lw = (base, 2.5) if on else (base, 1.2)
        else:
            f = FEATURES[t.feature[n]]
            thr = t.threshold[n]
            if f == "log_emissions":
                txt = (f"emissions \u2264\n{np.expm1(thr):,.0f} t/yr\n"
                       f"n={t.n_node_samples[n]}")
            elif f == "naics_sector_encoded":
                if encoder_kind(le) == "target":
                    txt = f"sector rate \u2264 {thr:.0%}\nn={t.n_node_samples[n]}"
                else:
                    txt = f"sector code \u2264 {thr:.0f}\nn={t.n_node_samples[n]}"
            else:
                txt = f"years \u2264 {thr:.0f}\nn={t.n_node_samples[n]}"
            if dark:
                fc, alpha = ("#232830", 1.0) if on else ("#16181d", 0.65)
            else:
                fc, alpha = ("#e8f1fd", 1.0) if on else ("#ffffff", 1.0)
            ec, lw = (ACCENT, 2.5) if on else (edge_dim, 1.2)
        box = FancyBboxPatch((x - 0.44, y - 0.34), 0.88, 0.68,
                             boxstyle="round,pad=0.02", facecolor=fc, alpha=alpha,
                             edgecolor=ec, linewidth=lw)
        ax.add_patch(box)
        ax.text(x, y, txt, ha="center", va="center", fontsize=9, color=INK,
                alpha=1.0 if on else 0.75, weight="bold" if on else "normal")
        for child, lab in ((t.children_left[n], "yes"), (t.children_right[n], "no")):
            if child != -1:
                cx, cy = pos[child]
                lit = on and child in path
                ax.annotate("", xy=(cx, cy + 0.34), xytext=(x, y - 0.34),
                            arrowprops=dict(arrowstyle="->",
                                            color=ACCENT if lit else edge_dim,
                                            lw=2 if lit else 1))
                ax.text((x + cx) / 2, (y + cy) / 2 + 0.05, lab, fontsize=8,
                        color=ACCENT if lit else GRAY, ha="center",
                        bbox=dict(facecolor=PAGE_BG, edgecolor="none", pad=1))
    ax.set_xlim(-0.9, counter[0] - 0.1)
    # Vertical room scales with tree depth instead of a hardcoded depth-3
    # window, so a retrained deeper tree still renders fully.
    ax.set_ylim(-(int(tree.max_depth) + 0.95), 0.75)
    ax.axis("off")
    fig.tight_layout()
    return fig


def tree_legend():
    """Plain-words key for the tree diagram labels."""
    q_bd = "#3a4048" if dark else "#c9ccd1"
    st.markdown(
        f"<div class='tlegend'>"
        f"<span><i class='sw' style='background:{PANEL_BG};"
        f"border:2px solid {ACCENT}'></i>path this facility took</span>"
        f"<span><i class='sw' style='background:{RED}33;"
        f"border:1px solid {RED}'></i><b>CCS Candidate</b>: pure CO2 stream, "
        f"suited to storage</span>"
        f"<span><i class='sw' style='background:{AMBER}33;"
        f"border:1px solid {AMBER}'></i><b>Potential CU Candidate</b>: mixed "
        f"stream, suited to utilization</span>"
        f"<span><i class='sw' style='background:{PANEL_BG};"
        f"border:1px solid {q_bd}'></i>a question the tree asks</span>"
        f"<span><b>n</b> = facilities that ended in that box</span>"
        f"</div>",
        unsafe_allow_html=True)


def pipeline_html():
    """The end-to-end flow as a visual strip of stages.

    Every number is computed from the loaded data and model, never
    hardcoded, so a retrain or data refresh cannot leave stale copy
    on screen.
    """
    counts = fac["priority"].value_counts()
    n_ccs = int(counts.get("CCS Candidate", 0))
    n_cu = int(counts.get("Potential CU Candidate", 0))
    stages = [
        (f"{len(yearly):,}", "yearly rows reported to ECCC"),
        ("Clean", "dedupe + standardize"),
        (f"{META['n_facilities_total']:,}", "facilities after averaging years"),
        (f"{len(fac)}", "above the 100k cutoff" if IS_V3 else "above the 100k TIER line"),
        ("Label", f"{CO2_CUTOFF:.0%} CO\u2082 rule \u2192 {n_ccs} CCS / {n_cu} CU"),
        (f"{len(FEATURES)}", "features: emissions, sector, years"),
        ("Tree", f"depth {tree.max_depth} \u2192 prediction + alerts"),
    ]
    cards = [f"<div class='pstep'><div class='pnum'>{n}</div>"
             f"<div class='plab'>{l}</div></div>" for n, l in stages]
    return "<div class='pipe'>" + "<div class='parrow'>\u2192</div>".join(cards) + "</div>"


def leaf_card(leaf, row, path):
    pred = row["model_prediction"]
    err = LEAF_ERR.get(leaf, 0.0)
    n = _LEAF_N.get(leaf, 0)
    priority_badge(pred)
    st.markdown(
        f"<div class='wleaf'><div style='color:{GRAY};font-size:14px;'>"
        f"This is the final group: "
        f"{n} facilities ended here, and the tree got "
        f"<b style='color:{INK};'>{err:.0%}</b> of them wrong in training. "
        f"That is where the alerts below come from.</div></div>",
        unsafe_allow_html=True)
    disagreement_banner(pred, row)
    borderline_banner(row)
    confidence_banner(pred, row)
    decision_box(row, leaf)
    st.pyplot(draw_tree_diagram(path))
    tree_legend()


# Preset walkthrough examples: a clean CCS case vs the trickiest CU case.
# Leaf ids come from the startup batch call (_startup_leaves), not from a
# per-row tree.apply: with the remote model each apply() is an HTTP POST,
# so the old loop fired one request per facility on every rerun.
def _walk_flags(r, leaf):
    return len(_flag_list(r, leaf))


_CLEAR_ID, _TRICKY_ID, _best = None, None, -1
for _pos, (_i, _r) in enumerate(fac.iterrows()):
    _lf = int(_startup_leaves[_pos])
    _p, _f = _r["model_prediction"], _walk_flags(_r, _lf)
    if _CLEAR_ID is None and _f == 0 and _p == "CCS Candidate":
        _CLEAR_ID = _r["facility_id"]
    if _p == "Potential CU Candidate" and _f > _best:
        _best, _TRICKY_ID = _f, _r["facility_id"]
_CLEAR_ID = _CLEAR_ID or fac["facility_id"].iloc[0]
_TRICKY_ID = _TRICKY_ID or fac["facility_id"].iloc[0]


st.markdown(f"""
<style>
.pipe{{display:flex;align-items:stretch;gap:4px;flex-wrap:wrap;margin:10px 0;}}
.pstep{{background:{PANEL_BG};border:1px solid {TRACK};border-radius:10px;
padding:10px 12px;min-width:105px;flex:1;}}
.pnum{{font-size:22px;font-weight:700;color:{ACCENT};}}
.plab{{font-size:12px;color:{GRAY};margin-top:2px;line-height:1.35;}}
.parrow{{align-self:center;color:{GRAY};font-size:16px;}}
.wstep{{background:{PANEL_BG};border-left:3px solid {ACCENT};
border-radius:0 10px 10px 0;padding:10px 14px;margin:8px 0;}}
.wstep-num{{font-size:11px;font-weight:700;color:{ACCENT};
text-transform:uppercase;letter-spacing:0.5px;}}
.wstep-q{{font-size:15px;font-weight:600;color:{INK};margin-top:2px;}}
.wstep-a{{font-size:14px;color:{GRAY};margin-top:2px;}}
.wstep-a b{{color:{INK};}}
.wgo{{color:{ACCENT};font-weight:700;}}
.wleaf{{background:{PANEL_BG};border:1px solid {TRACK};
border-radius:10px;padding:14px 16px;margin:12px 0;}}
.tlegend{{display:flex;flex-wrap:wrap;gap:8px 18px;margin:8px 0 4px;
font-size:12.5px;color:{GRAY};}}
.tlegend .sw{{display:inline-block;width:14px;height:14px;border-radius:4px;
margin-right:6px;vertical-align:-2px;}}
.tlegend b{{color:{INK};}}
.dbox{{background:{PANEL_BG};border:1px solid;border-radius:10px;
padding:14px 16px;margin:12px 0;}}
.dbox-title{{font-weight:700;font-size:16px;}}
.dbox-body{{color:{GRAY};font-size:14px;margin-top:6px;}}
.dbox ul{{color:{GRAY};font-size:14px;margin:8px 0 0;padding-left:20px;}}
.dbox li{{margin:4px 0;}}
.dbox b{{color:{INK};}}
</style>
""", unsafe_allow_html=True)


tab_about, tab_how, tab_screen, tab_facilities = st.tabs(
    ["About", "How it works", "Screening", "Facilities"])

with tab_about:
    st.subheader("What this is")
    if IS_V3:
        st.write("Carbon capture can trap CO2 before it reaches the air, but "
                 "it is expensive and cannot go everywhere. The real question "
                 "is simple. Where should it go first.")
        st.write("This tool answers that as a first pass. It reads public "
                 f"emissions data for {META['n_facilities_total']:,} Canadian "
                 "industrial facilities and screens the best candidates for "
                 "carbon capture or carbon utilization. What used to take "
                 "months of manual review now takes minutes, across every "
                 "facility, not just the big names.")
        st.write("Built as an AMII capstone project, extended to national "
                 "screening.")
    else:
        st.write("Alberta releases about 270 megatonnes of CO2 every year. Carbon "
                 "capture can trap it before it reaches the air, but it is expensive "
                 "and cannot go everywhere. The real question is simple. Where should "
                 "it go first.")
        st.write("This tool answers that as a first pass. It reads public emissions "
                 "data for 1,199 Alberta industrial facilities and screens the best "
                 "candidates for carbon capture or carbon utilization. What used to "
                 "take months of manual review now takes minutes, across every "
                 "facility, not just the big names.")
        st.write("Built as an AMII capstone project.")
    st.write("The visual version lives on the **How it works** tab: the full "
             "pipeline as a flow diagram, the tree itself, and an animated "
             "walkthrough of a facility going through it, question by question.")
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
    st.subheader("Download the data")
    st.write("The full datasets, so anyone can check the work or redo it from scratch.")
    d1, d2 = st.columns(2)
    if IS_V3:
        with d1:
            st.download_button("Cleaned national dataset (CSV)",
                               _file_bytes("data/Capstone_Dataset_clean_national.csv"),
                               file_name="Capstone_Dataset_clean_national.csv",
                               mime="text/csv",
                               use_container_width=True)
            st.caption(f"{len(yearly):,} yearly records from {META['n_facilities_total']:,} "
                       f"facilities, {int(yearly['year'].min())} to {int(yearly['year'].max())}. "
                       "The exact input the v3 model trained on.")
        with d2:
            st.download_button("Labeled national facilities (CSV)",
                               _file_bytes("data/facility_labeled_national.csv"),
                               file_name="facility_labeled_national.csv",
                               mime="text/csv",
                               use_container_width=True)
            st.caption(f"{len(fac)} above-threshold facilities with rule labels, "
                       "panel votes, and verdict tiers.")
    else:
        with d1:
            st.download_button("AMII dataset, as provided (CSV)",
                               _file_bytes("data/Capstone_Dataset.csv"),
                               file_name="Capstone_Dataset.csv",
                               mime="text/csv",
                               use_container_width=True)
            st.caption("18,772 yearly records from the AMII course package, before "
                       "any cleaning. The underlying source is ECCC's Greenhouse "
                       "Gas Reporting Program [1].")
        with d2:
            st.download_button("Cleaned dataset (CSV)",
                               _file_bytes("data/Capstone_Dataset_clean.csv"),
                               file_name="Capstone_Dataset_clean.csv",
                               mime="text/csv",
                               use_container_width=True)
            st.caption("6,999 rows after cleaning, the exact input the model "
                       "trained on: Alberta-only filter, bilingual headers "
                       "renamed, numeric types fixed, missing gas values set to "
                       "zero, duplicates removed.")
    st.subheader("How the data was cleaned")
    if IS_V3:
        _df = _v3_data_facts()
        st.write(f"**How messy it was.** The national file arrived with "
                 f"{_df['raw_rows']:,} yearly records across "
                 f"{_df['raw_facilities']:,} facilities and 13 provinces and "
                 f"territories, every header written in English and French, "
                 f"covering {_df['year_min']} to {_df['year_max']} under "
                 f"changing reporting rules. {_df['npri_missing']:,} cells "
                 f"were empty, all of them NPRI ids. The good news: zero "
                 f"duplicated facility-year rows.")
        st.write("**What we did.** Renamed the bilingual headers to short "
                 "English names. Forced the emissions columns to numeric types "
                 "so bad values could not sneak through. Grouped by GHGRP "
                 "facility ID, not by name, because facilities get renamed "
                 "across the years.")
        st.write(f"**The result.** {_df['clean_rows']:,} clean rows, "
                 f"{_df['clean_facilities']:,} facilities, years "
                 f"{_df['year_min']} to {_df['year_max']}, zero missing values "
                 f"in every column the model touches.")
        st.write("**Limitations.** Three honest ones. The 100,000 tonne cutoff "
                 "is a screening heuristic applied uniformly across provinces. "
                 "It is not a regulatory threshold outside Alberta, so treat it "
                 "as a triage line, not a legal one. Reporting rules changed "
                 f"over the {_df['n_years']} year window, so early years are "
                 f"thinner. And the "
                 "starting file is the AMII course extract, not ECCC's raw "
                 "publication, so the true original lives with the government "
                 "source [1].")
    else:
        st.write("**How messy it was.** The file arrived with 18,772 yearly records "
             "and 17 columns, every header written in English and French, some "
             "over 100 characters long. 10,704 cells were empty, worst in "
             "company trade names (8,296 missing) and cities (1,866). It "
             "covered all of Canada across 20 years of changing reporting "
             "rules, with 164 rows missing methane figures and 263 missing "
             "nitrous oxide. The good news: zero duplicated rows and zero "
             "zero-emission rows.")
        st.write("**What we did.** Renamed all 17 bilingual headers to short "
                 "English names. Kept Alberta only, which cut the file to 6,999 "
                 "rows. Forced 8 columns to numeric types so bad values could not "
                 "sneak through. Filled missing gas figures with zero on the "
                 "documented assumption that not reported means none reported, "
                 "set missing cities to Unknown, and filled trade names from "
                 "legal names. Removed duplicates and zero-emission rows (none "
                 "were found, the check stays as a guard). Grouped by GHGRP "
                 "facility ID, not by name, because 232 facilities were renamed "
                 "across the years.")
        st.write("**The result.** 6,999 clean rows, 1,199 facilities, years 2004 "
                 "to 2023, zero missing values in every column the model touches.")
        st.write("**Limitations.** Four honest ones. Missing gas set to zero is "
                 "an assumption, and it slightly inflates CO2 share for those "
                 "rows. The 2004 reporting methodology differs (in 60 rows the "
                 "reported total does not match the sum of the gases), but "
                 "rerunning the screen without those rows changes no outcome. "
                 "This covers Alberta only. And the starting file is the AMII "
                 "course extract, not ECCC's raw publication, so the true "
                 "original lives with the government source [1].")
    references()

with tab_how:
    st.subheader("The model, visually")
    st.write("This is the part that makes it a model and not a filter. Raw data "
             "flows through a pipeline, and every facility walks down a decision "
             "tree: three questions, asked in order, each answer choosing a "
             "branch until the facility lands on a verdict.")
    st.subheader("The full flow")
    st.markdown(pipeline_html(), unsafe_allow_html=True)
    st.caption("Gas shares never enter the features. They define the label, "
               "so the tree is not allowed to see them.")
    st.subheader("The tree itself")
    st.pyplot(draw_tree_diagram())
    tree_legend()
    _n_nodes = int(tree.tree_.node_count)
    _n_leaves = int((tree.tree_.children_left == -1).sum())
    st.caption(f"{_n_nodes} nodes, {_n_leaves} leaves. Every facility starts "
               f"at the top and answers its way down.")
    st.subheader("Watch a facility go through it")
    st.write("Two presets, or pick any facility yourself.")
    e1, e2 = st.columns(2)
    with e1:
        if st.button("Show a clear-cut case", use_container_width=True):
            st.session_state["walk_pick"] = _CLEAR_ID
            st.session_state["walk_autorun"] = True
    with e2:
        if st.button("Show a tricky case", use_container_width=True):
            st.session_state["walk_pick"] = _TRICKY_ID
            st.session_state["walk_autorun"] = True
    wids = sorted(fac["facility_id"].unique(), key=lambda i: display[i].lower())
    wdefault = next((k for k, i in enumerate(wids) if "Alberta-Pacific" in display[i]), 0)
    wchoice = st.selectbox("Facility", wids, index=wdefault, key="walk_pick",
                           format_func=lambda i: display[i])
    wrow = fac[fac["facility_id"] == wchoice].iloc[0]
    run_clicked = st.button("\u25b6 Run the flow", type="primary")
    if run_clicked or st.session_state.pop("walk_autorun", False):
        wcards, wleaf, wpath = story_cards(wrow)
        st.session_state.walk = {"id": wchoice, "cards": wcards,
                                 "leaf": wleaf, "path": wpath}
        box = st.empty()
        for i in range(len(wcards)):
            with box.container():
                for j in range(i + 1):
                    reason_card(wcards[j][0], wcards[j][1], j + 1, len(wcards))
            time.sleep(0.8)
        box.empty()
        for j in range(len(wcards)):
            reason_card(wcards[j][0], wcards[j][1], j + 1, len(wcards))
        leaf_card(wleaf, wrow, wpath)
    elif st.session_state.get("walk", {}).get("id") == wchoice:
        wk = st.session_state.walk
        for j in range(len(wk["cards"])):
            reason_card(wk["cards"][j][0], wk["cards"][j][1], j + 1, len(wk["cards"]))
        leaf_card(wk["leaf"], wrow, wk["path"])

with tab_screen:
    _region = "a Canadian" if IS_V3 else "an Alberta"
    st.write(f"Pick {_region} industrial facility to see whether it screens as a "
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

    pred = row["model_prediction"]
    priority_badge(pred)
    disagreement_banner(pred, row)
    borderline_banner(row)
    confidence_banner(pred, row)
    st.divider()
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader(display[choice])
        st.write(f"**Sector:** {row['sector']}")
        st.write(f"**Average annual emissions:** {row['avg_annual_emissions']:,.0f} tonnes CO2e")
        st.write(f"**CO2 share of emissions:** {row['co2_share']:.0%}")
        st.write(f"**Years reported:** {int(row['years_reported'])}")
        if int(row['years_reported']) <= 5:
            st.caption(f"Only {int(row['years_reported'])} years of data: "
                       f"averages are less stable.")
        st.write(f"**{'Threshold' if IS_V3 else 'TIER'} band:** {row['emission_band']}")
    with right:
        st.subheader("At a glance")
        share_bar(row["co2_share"])
        peers = fac[fac["sector"] == row["sector"]]
        if len(peers) <= 1:
            st.caption("Only facility in this sector: no peer comparison available.")
        else:
            peer_bar(row["avg_annual_emissions"],
                     peers["avg_annual_emissions"].median(), row["sector"])
    st.subheader("Why this result")
    for i, s in enumerate(explain_path(row), 1):
        st.write(f"{i}. {s}")
    with st.expander("About this model"):
        if IS_V3:
            _df2 = _v3_data_facts()
            st.write(f"Decision Tree classifier, maximum depth 3, trained on "
                     f"{_df2['n_above_threshold']} "
                     f"above-threshold Canadian facilities. Features: log-scaled average "
                     "emissions, target-encoded industry sector, years reported. Gas shares were "
                     "excluded from features to avoid label leakage. Evaluated with "
                     "5-fold stratified cross validation: accuracy 0.80, macro F1 0.74. "
                     "A panel of four models votes on every facility: unanimous and "
                     "majority verdicts are the steadiest, contested ones need human review.")
        else:
            st.write("Decision Tree classifier, maximum depth 3, trained on 150 "
                     "above-threshold Alberta facilities [2]. Features: log-scaled average "
                     "emissions, encoded industry sector, years reported. Gas shares were "
                     "excluded from features to avoid label leakage. Evaluated with "
                     "5-fold stratified cross validation: accuracy 0.75, macro F1 0.62. "
                     "CU precision is 0.30, so most CU predictions are actually CCS "
                     "facilities: treat every CU flag as needing human review.")
        if IS_V3:
            st.write(f"Data: Environment and Climate Change Canada, Greenhouse Gas "
                     f"Reporting Program, public dataset {_df2['year_min']} to "
                     f"{_df2['year_max']} [1].")
        else:
            st.write("Data: Environment and Climate Change Canada, Greenhouse Gas "
                     "Reporting Program, public dataset 2004 to 2023 [1].")
    st.divider()
    tier_badge(row)
    y = yearly[yearly["facility_id"] == choice].sort_values("year")
    latest = y.iloc[-1] if len(y) else None

    def _screen_val(col):
        if latest is not None and col in y.columns and pd.notna(latest[col]):
            return latest[col]
        return "Not reported"

    screen_reasons = [str(row[c]).strip()
                      for c in ("reason_1", "reason_2", "reason_3")
                      if c in row.index and pd.notna(row[c])
                      and str(row[c]).strip()]
    if IS_V3:
        _econ_summary(row)
        _econ_charts(row)
        _assumptions_panel()
    _memo_section(row, _screen_val, screen_reasons)

with tab_facilities:
    st.write("Every screened facility, with its data source, the rule-based "
             "result, and the model prediction side by side. All figures "
             "come from Environment and Climate Change Canada's Greenhouse Gas "
             "Reporting Program [1]. " + (
                 "Facilities averaging at least 100,000 tonnes CO2e per year "
                 "form the national screening population: a screening cutoff, "
                 "not a regulatory threshold outside Alberta. "
                 if IS_V3 else
                 "Facilities averaging at least 100,000 tonnes CO2e per year "
                 "fall under Alberta's TIER regulation [2] and form the "
                 "screening population. ") +
             "Select a row for the full facility profile.")
    f1, f2, f3 = st.columns(3)
    with f1:
        q = st.text_input("Search by name")
    with f2:
        sectors = st.multiselect("Sector", sorted(fac["sector"].unique()),
                                 default=sorted(fac["sector"].unique()))
    with f3:
        prios = st.multiselect("Rule-based result", ["CCS Candidate", "Potential CU Candidate"],
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
    # Display-ready copy: numbers are formatted as text here because
    # column_config number formatting did not render on the live app.
    # t keeps the raw numbers for sorting and row selection.
    disp = t[["facility_name", "sector", "avg_annual_emissions", "co2_share",
              "priority", "model_prediction", "source"]].copy()
    disp["avg_annual_emissions"] = disp["avg_annual_emissions"].map(lambda x: f"{x:,.0f}")
    disp["co2_share"] = disp["co2_share"].map(lambda x: f"{x:.0%}")
    event = st.dataframe(
        disp,
        column_config={
            "facility_name": st.column_config.TextColumn("Facility"),
            "sector": st.column_config.TextColumn("Sector"),
            "avg_annual_emissions": st.column_config.TextColumn("Avg emissions (tCO2e/yr)"),
            "co2_share": st.column_config.TextColumn("CO2 share"),
            "priority": st.column_config.TextColumn("Rule-based result"),
            "model_prediction": st.column_config.TextColumn("Model prediction"),
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

# Phone layout: stack side-by-side columns vertically on narrow screens.
# Plain (non-f) string on purpose: no brace escaping to get wrong.
st.markdown("""
<style>
@media (max-width: 640px) {
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    flex: 1 1 100% !important; min-width: 100% !important; }
  .block-container { padding-left: 1rem !important;
    padding-right: 1rem !important; }
}
</style>
""", unsafe_allow_html=True)
