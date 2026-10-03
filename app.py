"""Capstone deployment app: CCS/CU priority screening for Alberta facilities."""
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
    if priority == "CCS Candidate":
        st.success("## CCS Candidate")
        st.write("High concentration CO2 stream. Suited to carbon capture and storage [3].")
    else:
        st.info("## Potential CU Candidate")
        st.write("Mixed gas stream. Better directed toward carbon utilization pathways [3].")


def facility_profile(row):
    """Full profile for one facility row."""
    st.subheader(row["facility_name"])
    a, b = st.columns(2)
    with a:
        st.write(f"**Sector:** {row['sector']}")
        st.write(f"**Company:** {row['company_legal']}")
        st.write(f"**City:** {row['city']}")
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


st.title("Carbon Capture Priority Screening")
tab_screen, tab_facilities = st.tabs(["Screening", "Facilities"])

with tab_screen:
    st.write("Pick an Alberta industrial facility to see whether it screens as a "
             "carbon capture candidate or a carbon utilization candidate, and why. "
             "Built on public emissions data [1].")
    names = sorted(fac["facility_name"].unique())
    choice = st.selectbox("Facility", names, key="screen_pick")
    row = fac[fac["facility_name"] == choice].iloc[0]
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
        x = row[FEATURES].values.reshape(1, -1)
        priority_badge(tree.predict(x)[0])
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

    t = fac[(fac["facility_name"].str.contains(q, case=False, na=False))
            & (fac["sector"].isin(sectors))
            & (fac["priority"].isin(prios))
            & (fac["avg_annual_emissions"] >= min_e)].copy()
    t["source"] = GHGRP_URL
    t = t.sort_values("avg_annual_emissions", ascending=False)
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
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        use_container_width=True,
    )
    rows = event.selection.rows
    if rows:
        sel = t.iloc[rows[0]]
        st.divider()
        facility_profile(fac[fac["facility_id"] == sel["facility_id"]].iloc[0])

    references()
