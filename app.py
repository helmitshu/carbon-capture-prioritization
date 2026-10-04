"""Capstone deployment app: CCS/CU priority screening for Alberta facilities."""
import hashlib
import re
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import streamlit as st

from features import CO2_CUTOFF

GHGRP_URL = "https://open.canada.ca/data/en/dataset/a8ba14b7-7f23-462a-bdbb-83b0ef629823"
TIER_URL = "https://www.alberta.ca/technology-innovation-and-emissions-reduction-regulation"

st.set_page_config(page_title="CCS Priority Screening", layout="wide")

bundle = joblib.load("model/model.pkl")
tree, le = bundle["model"], bundle["encoder"]
FEATURES = bundle["features"]
fac = pd.read_csv("model/facilities.csv")
yearly = pd.read_csv("data/Capstone_Dataset_clean.csv")
# Model prediction for every facility, computed once. The Facilities table
# shows this next to the rule-based label so the two never get mixed up.
fac["model_prediction"] = tree.predict(fac[FEATURES].values)

# Per-leaf model reliability, computed once from in-sample fit. Leaves where
# the tree is frequently wrong get their own alert. (Raw leaf probabilities
# are distorted by balanced class weights, so the error rate is the honest
# signal, not the probability.)
_Xall = fac[FEATURES].values
_leaf_all = tree.apply(_Xall)
_pred_all = tree.predict(_Xall)
_true_all = fac["priority"].values
LEAF_ERR = {}
for lf in set(_leaf_all):
    m = _leaf_all == lf
    LEAF_ERR[int(lf)] = float((_pred_all[m] != _true_all[m]).mean())
HIGH_ERR_THRESHOLD = 0.25
BORDERLINE_PTS = 0.03  # CO2 share within 3 points of the 85% cutoff

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
    """Flag facilities whose CO2 share sits near the arbitrary 85% cutoff."""
    gap = abs(row["co2_share"] - 0.85)
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
    lf = int(tree.apply(row[FEATURES].values.reshape(1, -1))[0])
    err = LEAF_ERR.get(lf, 0.0)
    if err <= HIGH_ERR_THRESHOLD:
        return
    _alert_box(
        "Low model confidence. Needs human review.",
        f"The model predicted <b style='color:{INK};'>{pred}</b>, but "
        f"facilities landing in this part of the tree were misclassified "
        f"{err:.0%} of the time in training. Treat the prediction with "
        f"extra skepticism and lean on the data.")


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
        st.write(f"**TIER band:** {row['emission_band']}")
    st.write("**Emissions history**")
    hist = yearly[yearly["facility_id"] == row["facility_id"]].sort_values("year")
    st.line_chart(hist.set_index("year")["total_emissions"])
    st.write("**Screening result**")
    pred = row["model_prediction"]
    priority_badge(pred)
    disagreement_banner(pred, row)
    borderline_banner(row)
    confidence_banner(pred, row)
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
            codes = le.transform(le.classes_)
            side = le.classes_[codes <= thr] if go_left else le.classes_[codes > thr]
            q = (f"Is the sector code \u2264 {thr:.0f}? "
                 f"({len(side)} sectors go {'left' if go_left else 'right'})")
            a = f"{row['sector']} (code {val:.0f})"
        else:
            q = f"Was it reported for \u2264 {thr:.0f} years?"
            a = f"{val:.0f} years"
        steps.append({"node": node, "q": q, "a": a, "left": go_left})
        node = t.children_left[node] if go_left else t.children_right[node]
    return steps, node


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
    fig, ax = plt.subplots(figsize=(12, 6.2))
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
            fc, alpha = base, 0.30 if on else 0.10
            ec, lw = (base, 2.5) if on else ("#3a4048", 1.0)
        else:
            f = FEATURES[t.feature[n]]
            thr = t.threshold[n]
            if f == "log_emissions":
                txt = (f"emissions \u2264\n{np.expm1(thr):,.0f} t/yr\n"
                       f"n={t.n_node_samples[n]}")
            elif f == "naics_sector_encoded":
                txt = f"sector code \u2264 {thr:.0f}\nn={t.n_node_samples[n]}"
            else:
                txt = f"years \u2264 {thr:.0f}\nn={t.n_node_samples[n]}"
            fc, alpha = ("#232830", 1.0) if on else ("#16181d", 0.65)
            ec, lw = (ACCENT, 2.5) if on else ("#3a4048", 1.0)
        box = FancyBboxPatch((x - 0.44, y - 0.34), 0.88, 0.68,
                             boxstyle="round,pad=0.02", facecolor=fc, alpha=alpha,
                             edgecolor=ec, linewidth=lw)
        ax.add_patch(box)
        ax.text(x, y, txt, ha="center", va="center", fontsize=8, color=INK,
                alpha=1.0 if on else 0.75, weight="bold" if on else "normal")
        for child, lab in ((t.children_left[n], "yes"), (t.children_right[n], "no")):
            if child != -1:
                cx, cy = pos[child]
                lit = on and child in path
                ax.annotate("", xy=(cx, cy + 0.34), xytext=(x, y - 0.34),
                            arrowprops=dict(arrowstyle="->",
                                            color=ACCENT if lit else "#3a4048",
                                            lw=2 if lit else 1))
                ax.text((x + cx) / 2, (y + cy) / 2 + 0.05, lab, fontsize=7,
                        color=ACCENT if lit else GRAY, ha="center",
                        bbox=dict(facecolor=PAGE_BG, edgecolor="none", pad=1))
    ax.set_xlim(-0.9, counter[0] - 0.1)
    ax.set_ylim(-3.95, 0.75)
    ax.axis("off")
    fig.tight_layout()
    return fig


def pipeline_html():
    """The end-to-end flow as a visual strip of stages."""
    stages = [
        ("6,999", "yearly rows reported to ECCC"),
        ("Clean", "dedupe + standardize"),
        ("1,199", "facilities after averaging years"),
        ("150", "above the 100k TIER line"),
        ("Label", "85% CO\u2082 rule \u2192 130 CCS / 20 CU"),
        ("3", "features: emissions, sector, years"),
        ("Tree", "depth 3 \u2192 prediction + alerts"),
    ]
    cards = [f"<div class='pstep'><div class='pnum'>{n}</div>"
             f"<div class='plab'>{l}</div></div>" for n, l in stages]
    return "<div class='pipe'>" + "<div class='parrow'>\u2192</div>".join(cards) + "</div>"


def step_card(s, i, row):
    verdict = "YES" if s["left"] else "NO"
    direction = "go left" if s["left"] else "go right"
    st.markdown(
        f"<div class='wstep'><div class='wstep-num'>Step {i + 1}</div>"
        f"<div class='wstep-q'>{s['q']}</div>"
        f"<div class='wstep-a'>{display[row['facility_id']]}: {s['a']} "
        f"\u2192 <b>{verdict}</b> <span class='wgo'>{direction}</span></div></div>",
        unsafe_allow_html=True)


def leaf_card(leaf, row, steps):
    pred = row["model_prediction"]
    err = LEAF_ERR.get(leaf, 0.0)
    n = _LEAF_N.get(leaf, 0)
    priority_badge(pred)
    st.markdown(
        f"<div class='wleaf'><div style='color:{GRAY};font-size:14px;'>"
        f"Landed on leaf <b style='color:{INK};'>{leaf}</b>: "
        f"{n} facilities took this path, and the tree got "
        f"<b style='color:{INK};'>{err:.0%}</b> of them wrong in training. "
        f"That is where the alerts below come from.</div></div>",
        unsafe_allow_html=True)
    disagreement_banner(pred, row)
    borderline_banner(row)
    confidence_banner(pred, row)
    st.pyplot(draw_tree_diagram([s["node"] for s in steps] + [leaf]))


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
</style>
""", unsafe_allow_html=True)


tab_about, tab_how, tab_screen, tab_facilities = st.tabs(
    ["About", "How it works", "Screening", "Facilities"])

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
    st.caption("11 nodes, 6 leaves. Every facility starts at the top and "
               "answers its way down.")
    st.subheader("Watch a facility go through it")
    wids = sorted(fac["facility_id"].unique(), key=lambda i: display[i].lower())
    wdefault = next((k for k, i in enumerate(wids) if "Alberta-Pacific" in display[i]), 0)
    wchoice = st.selectbox("Facility", wids, index=wdefault, key="walk_pick",
                           format_func=lambda i: display[i])
    wrow = fac[fac["facility_id"] == wchoice].iloc[0]
    if st.button("\u25b6 Run the flow", type="primary"):
        wsteps, wleaf = tree_walk(wrow)
        st.session_state.walk = {"id": wchoice, "steps": wsteps, "leaf": wleaf}
        box = st.empty()
        for i, s in enumerate(wsteps):
            with box.container():
                for j in range(i):
                    step_card(wsteps[j], j, wrow)
                step_card(s, i, wrow)
            time.sleep(0.8)
        box.empty()
        for j, s in enumerate(wsteps):
            step_card(s, j, wrow)
        leaf_card(wleaf, wrow, wsteps)
    elif st.session_state.get("walk", {}).get("id") == wchoice:
        wk = st.session_state.walk
        for j, s in enumerate(wk["steps"]):
            step_card(s, j, wrow)
        leaf_card(wk["leaf"], wrow, wk["steps"])

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
        st.write(f"**TIER band:** {row['emission_band']}")
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
        st.write("Decision Tree classifier, maximum depth 3, trained on 150 "
                 "above-threshold Alberta facilities [2]. Features: log-scaled average "
                 "emissions, encoded industry sector, years reported. Gas shares were "
                 "excluded from features to avoid label leakage. Evaluated with "
                 "5-fold stratified cross validation: accuracy 0.75, macro F1 0.62. "
                 "CU precision is 0.30, so most CU predictions are actually CCS "
                 "facilities: treat every CU flag as needing human review.")
        st.write("Data: Environment and Climate Change Canada, Greenhouse Gas "
                 "Reporting Program, public dataset 2004 to 2023 [1].")

with tab_facilities:
    st.write("Every screened facility, with its data source, the rule-based "
             "result, and the model prediction side by side. All figures "
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
    event = st.dataframe(
        t[["facility_name", "sector", "avg_annual_emissions", "co2_share",
           "priority", "model_prediction", "source"]],
        column_config={
            "facility_name": st.column_config.TextColumn("Facility"),
            "sector": st.column_config.TextColumn("Sector"),
            "avg_annual_emissions": st.column_config.NumberColumn(
                "Avg emissions (tCO2e/yr)", format="%.0f"),
            "co2_share": st.column_config.NumberColumn("CO2 share", format="%.0%%"),
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
