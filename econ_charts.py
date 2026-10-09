"""Matplotlib figure builders for the v3 economics visuals.

No Streamlit here: each function takes facility inputs and returns a
matplotlib Figure. The app renders them with st.pyplot. All numbers
come from economics.econ_chart_data, the same source as the text
figures, so charts can never disagree with the copy.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from economics import econ_chart_data

_TEAL = "#0f766e"
_INK = "#0f172a"
_GRAY = "#cbd5e1"
_AMBER = "#b45309"


def _co2_share_text(co2_share: float) -> str:
    return f"{co2_share:.0%}"


def emissions_mix_donut(avg_annual_emissions: float, sector: str,
                       co2_share: float = 1.0):
    """Donut of the emissions mix in CO2e tonnes: CO2 vs other GHGs."""
    data = econ_chart_data(avg_annual_emissions, sector,
                          co2_share=co2_share)
    mix = data["mix"]
    total_kt = data["total_co2e"] / 1e3
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    vals = [m["co2e"] for m in mix]
    wedges, _ = ax.pie(vals, colors=[_TEAL, _GRAY], startangle=90,
                       counterclock=False,
                       wedgeprops=dict(width=0.45, edgecolor="white"))
    ax.text(0, 0, f"{total_kt:,.0f} kt\nCO2e", ha="center", va="center",
            fontsize=11, fontweight="bold", color=_INK)
    ax.set_title("Emissions mix", fontsize=12, fontweight="bold",
                 color=_INK)
    labels = [f"{m['label']}: {m['co2e'] / 1e3:,.0f} kt "
              f"({_co2_share_text(m['share'])})" for m in mix]
    ax.legend(wedges, labels, loc="center", bbox_to_anchor=(0.5, -0.12),
              fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


def liability_split_bar(avg_annual_emissions: float, sector: str,
                        co2_share: float = 1.0):
    """Stacked bar of the 2030 carbon bill: abatable vs locked in."""
    data = econ_chart_data(avg_annual_emissions, sector,
                          co2_share=co2_share)
    liab = data["liability"]
    ab_m = liab["abatable"] / 1e6
    li_m = liab["locked_in"] / 1e6
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.barh(["2030 carbon bill"], [ab_m], color=_TEAL)
    ax.barh(["2030 carbon bill"], [li_m], left=[ab_m], color=_GRAY)
    ax.set_xlabel("$M per year", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend([f"Abatable by capture: ${ab_m:,.1f}M",
               f"Locked in (other gases): ${li_m:,.1f}M"],
              fontsize=9, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(bottom=0.30, top=0.94, left=0.22, right=0.96)
    return fig


def cost_band_chart(avg_annual_emissions: float, sector: str,
                    co2_share: float = 1.0):
    """Low/base/high capture cost per tonne against the carbon price."""
    data = econ_chart_data(avg_annual_emissions, sector,
                          co2_share=co2_share)
    cb = data["cost_band"]
    fig, ax = plt.subplots(figsize=(5.2, 2.0))
    cases = [("Low", cb["low"]), ("Base", cb["base"]),
             ("High", cb["high"])]
    for i, (label, val) in enumerate(cases):
        ax.scatter([val], [label], s=90, color=_TEAL, zorder=3)
        ax.text(val, i, f"  ${val:,.0f}/t", va="center", fontsize=9,
                color=_INK)
    ax.axvline(cb["carbon_price"], color=_AMBER, linestyle="--",
               linewidth=1.5)
    ax.text(cb["carbon_price"], 2.62,
            f"2030 carbon price ${cb['carbon_price']:,.0f}/t",
            ha="center", fontsize=8, color=_AMBER)
    ax.set_xlabel("CAD per tonne of CO2", fontsize=9)
    ax.set_ylim(-0.6, 2.9)
    ax.tick_params(labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def emissions_trend_chart(facility_id: str, years: int = 20,
                          figsize=(8.5, 3.8), show_title: bool = True):
    """Annual reported emissions bars plus a trend line, last N years.

    One value per reported year, so bars are the honest format
    (candlesticks need intra-period ranges the records do not have).
    Real records only; no projection here.
    """
    import numpy as np
    import pandas as pd

    # Same yearly source the v3 app reads; fall back to the older file.
    yearly_path = "data/Capstone_Dataset_clean_national.csv"
    try:
        y = pd.read_csv(yearly_path)
    except FileNotFoundError:
        y = pd.read_csv("data/Capstone_Dataset_clean.csv")
    g = y[y["facility_id"] == str(facility_id)].sort_values("year")
    hist = g[["year", "total_emissions"]].dropna().tail(years)
    if len(hist) < 2:
        raise ValueError(f"not enough history for {facility_id}")
    yrs = hist["year"].to_numpy()
    mt = hist["total_emissions"].to_numpy() / 1e6
    name = str(g.iloc[0]["facility_name"])
    try:
        op = str(g.iloc[0]["company_trade"])
    except Exception:
        op = ""

    coef = np.polyfit(yrs, mt, 1)
    trend = np.polyval(coef, yrs)
    direction = "rising" if coef[0] > 0.02 else (
        "falling" if coef[0] < -0.02 else "flat")

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(yrs, mt, color=_TEAL, width=0.7, zorder=3)
    ax.plot(yrs, trend, color=_AMBER, linestyle="--", linewidth=1.8,
            label=f"Trend ({direction})")
    if show_title:
        ax.set_title(f"{name}", fontsize=12, fontweight="bold", color=_INK,
                     loc="left", pad=26)
        ax.text(0, 1.015,
                f"{op}  |  {len(hist)} years of reported emissions" if op
                else f"{len(hist)} years of reported emissions",
                transform=ax.transAxes, fontsize=9, color="#64748b",
                va="bottom", ha="left")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Mt CO2e per year", fontsize=10)
    ax.set_xticks(yrs[::2].astype(int))
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.tick_params(labelsize=9)
    if not show_title:
        # Compact mode for the one-page memo: smaller type, no x label.
        ax.set_xlabel("")
        ax.set_ylabel("Mt CO2e/yr", fontsize=8)
        ax.tick_params(labelsize=7)
        for t in ax.get_legend().get_texts():
            t.set_fontsize(7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
