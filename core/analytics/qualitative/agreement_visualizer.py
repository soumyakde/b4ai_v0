"""
core/analytics/qualitative/agreement_visualizer.py
-------------------------------------------------------
Visualization layer for Agreement Analytics

Produces publication-ready diagnostic plots:
    • ICC bar chart
    • Bland–Altman agreement plot
    • Construct correlation heatmap

NO STREAMLIT IMPORTS
-------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# ICC BAR CHART
# =====================================================

def plot_icc_summary(agreement_df: pd.DataFrame):
    """
    Bar chart of ICC per construct.
    """

    if agreement_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))

    agreement_df = agreement_df.sort_values("ICC2_1")

    ax.barh(
        agreement_df["construct"],
        agreement_df["ICC2_1"],
    )

    ax.set_xlabel("ICC(2,1)")
    ax.set_title("Human–LLM Agreement by Construct")

    ax.axvline(0.5, linestyle="--")
    ax.axvline(0.75, linestyle="--")
    ax.axvline(0.9, linestyle="--")

    ax.set_xlim(0, 1)

    plt.tight_layout()
    return fig


# =====================================================
# BLAND–ALTMAN PLOT
# =====================================================

def plot_bland_altman(aligned_df: pd.DataFrame, construct: str):
    """
    Agreement diagnostic for one construct.
    """

    df = aligned_df[aligned_df["construct"] == construct]

    pivot = df.pivot(
        index="target_id",
        columns="rater",
        values="rating",
    ).dropna()

    if pivot.empty:
        return None

    human = pivot["Human"]
    llm = pivot["LLM"]

    mean_scores = (human + llm) / 2
    diff_scores = llm - human

    mean_diff = diff_scores.mean()
    sd_diff = diff_scores.std()

    upper = mean_diff + 1.96 * sd_diff
    lower = mean_diff - 1.96 * sd_diff

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(mean_scores, diff_scores, alpha=0.6)

    ax.axhline(mean_diff, linestyle="--")
    ax.axhline(upper, linestyle=":")
    ax.axhline(lower, linestyle=":")

    ax.set_xlabel("Mean Rating (Human & LLM)")
    ax.set_ylabel("Difference (LLM − Human)")
    ax.set_title(f"Bland–Altman Plot — {construct}")

    plt.tight_layout()
    return fig


# =====================================================
# CORRELATION HEATMAP
# =====================================================

def plot_construct_correlation(aligned_df: pd.DataFrame):
    """
    Heatmap of construct correlations (LLM ratings).
    Useful for construct validity inspection.
    """

    llm = aligned_df[aligned_df["rater"] == "LLM"]

    pivot = llm.pivot_table(
        index="target_id",
        columns="construct",
        values="rating",
    )

    if pivot.shape[1] < 2:
        return None

    corr = pivot.corr()

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        square=True,
        ax=ax,
    )

    ax.set_title("Construct Correlation (LLM Ratings)")

    plt.tight_layout()
    return fig