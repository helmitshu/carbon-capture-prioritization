"""Capstone deployment app: screen an Alberta facility for CCS/CU priority."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CCS Priority Screening", layout="wide")

bundle = joblib.load("model/model.pkl")
tree, le = bundle["model"], bundle["encoder"]
FEATURES = bundle["features"]
fac = pd.read_csv("model/facilities.csv")


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


st.title("Carbon Capture Priority Screening")
st.write("Pick an Alberta industrial facility to see whether it screens as a "
         "carbon capture candidate or a carbon utilization candidate, and why. "
         "Built on public emissions data.")

names = sorted(fac["facility_name"].unique())
choice = st.selectbox("Facility", names)
row = fac[fac["facility_name"] == choice].iloc[0]
# the screening result is the MODEL's prediction, not the stored label
x = row[FEATURES].values.reshape(1, -1)
priority = tree.predict(x)[0]
steps = explain_path(row)

left, right = st.columns([1, 1.4])
with left:
    st.subheader(choice)
    st.write(f"**Sector:** {row['sector']}")
    st.write(f"**Average annual emissions:** {row['avg_annual_emissions']:,.0f} tonnes CO2e")
    st.write(f"**CO2 share of emissions:** {row['co2_share']:.0%}")
    st.write(f"**Years reported:** {int(row['years_reported'])}")
    st.write(f"**TIER band:** {row['emission_band']}")
with right:
    st.subheader("Screening result")
    if priority == "CCS Candidate":
        st.success("## CCS Candidate")
        st.write("High concentration CO2 stream. Suited to carbon capture and storage.")
    else:
        st.info("## Potential CU Candidate")
        st.write("Mixed gas stream. Better directed toward carbon utilization pathways.")
    st.subheader("Why this result")
    for i, s in enumerate(steps, 1):
        st.write(f"{i}. {s}")

st.divider()
st.subheader("Priority candidates by sector")
sec = fac.groupby("sector").agg(
    facilities=("facility_id", "count"),
    ccs=("priority", lambda s: int((s == "CCS Candidate").sum())),
).sort_values("ccs", ascending=False).head(10)
st.bar_chart(sec["ccs"])

with st.expander("About this model"):
    st.write("Decision Tree classifier, maximum depth 3, trained on 150 "
             "above-threshold Alberta facilities. Features: log-scaled average "
             "emissions, encoded industry sector, years reported. Evaluated with "
             "5-fold stratified cross validation: accuracy 0.75, macro F1 0.62. "
             "Gas shares were excluded from features to avoid label leakage.")
    st.write("Data: Environment and Climate Change Canada, Greenhouse Gas "
             "Reporting Program, public dataset 2004 to 2023.")
