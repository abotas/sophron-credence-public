"""Exp 4: Test-Retest Reliability tab."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from scipy import stats as sp_stats

from credence.viz.constants import (
    BUCKET_LABELS,
    BUCKET_ORDER,
    CATEGORY_COLORS,
    TEST_RETEST_MAD_TARGET,
    TEST_RETEST_SPEARMAN_TARGET,
)
from credence.viz.data import load_calibration
from credence.viz.formatting import short_model
from credence.viz.stats import bootstrap_mean_ci


def render() -> None:
    """Render the test-retest reliability tab."""
    st.subheader("V3 Test-Retest: Do repeated pipeline runs yield similar estimates?")
    st.caption(
        "We run the full pipeline twice over the 100 calibration propositions. Each run "
        "regenerates prompts end-to-end; target-model responses and judge evaluations "
        "are also re-produced. Comparing median credences across runs tests measurement stability."
    )

    all_cal = load_calibration()
    if all_cal.is_empty() or "run_id" not in all_cal.columns:
        st.info("Test-retest requires two calibration runs (run_id 0 and 1).")
        return

    run_ids = sorted(all_cal["run_id"].unique().to_list())
    if len(run_ids) < 2:
        st.info(f"Only {len(run_ids)} run(s) found. Test-retest requires at least 2.")
        return

    run0 = all_cal.filter(pl.col("run_id") == run_ids[0])
    run1 = all_cal.filter(pl.col("run_id") == run_ids[1])

    # Compute proposition-level medians for each run
    paired = _build_paired_data(run0, run1)
    if paired.is_empty():
        st.warning("No overlapping propositions between the two runs.")
        return

    target_models = sorted(paired["target_model"].unique().to_list())
    if len(target_models) > 1:
        selected_model = st.selectbox("Target model", target_models, key="tr_model")
        model_df = paired.filter(pl.col("target_model") == selected_model)
    else:
        model_df = paired

    # 1. Run 1 vs Run 2 scatter
    st.subheader("Run 1 vs Run 2 Median Credences")
    _render_scatter(model_df)

    # 2. Per-category MAD bar chart
    st.subheader("Mean Absolute Difference by Category")
    _render_mad_chart(model_df)

    # 3. Correlation summary table
    st.subheader("Correlation Summary")
    _render_correlation_table(model_df)


def _build_paired_data(run0: pl.DataFrame, run1: pl.DataFrame) -> pl.DataFrame:
    """Build paired dataset with proposition-level medians from both runs."""
    # Compute medians per (proposition, category, target_model) for each run
    medians0 = (
        run0.filter(pl.col("consensus_credence").is_not_null())
        .group_by(["proposition", "category", "target_model"])
        .agg(pl.col("consensus_credence").median().alias("run0_median"))
    )
    medians1 = (
        run1.filter(pl.col("consensus_credence").is_not_null())
        .group_by(["proposition", "category", "target_model"])
        .agg(pl.col("consensus_credence").median().alias("run1_median"))
    )

    return medians0.join(medians1, on=["proposition", "category", "target_model"])


def _compute_correlations(
    run0_vals: list[float],
    run1_vals: list[float],
) -> dict[str, float]:
    """Compute Pearson r, Spearman r, and MAD between two lists."""
    arr0 = np.array(run0_vals)
    arr1 = np.array(run1_vals)

    pearson_r = float(sp_stats.pearsonr(arr0, arr1).statistic)
    spearman_r = float(sp_stats.spearmanr(arr0, arr1).statistic)
    mad = float(np.mean(np.abs(arr0 - arr1)))

    return {"pearson_r": pearson_r, "spearman_r": spearman_r, "mad": mad, "n": len(arr0)}


def _render_correlation_table(df: pl.DataFrame) -> None:
    """Render table of correlation metrics per category and overall."""
    rows = []

    # Per category
    for bucket in BUCKET_ORDER:
        bucket_df = df.filter(pl.col("category") == bucket)
        if bucket_df.height < 3:
            rows.append({
                "Category": BUCKET_LABELS[bucket],
                "Pearson r": "—",
                "Spearman r": "—",
                "MAD": "—",
                "n": bucket_df.height,
                "Status": "—",
            })
            continue

        r0 = bucket_df["run0_median"].to_list()
        r1 = bucket_df["run1_median"].to_list()
        corrs = _compute_correlations(r0, r1)

        sp_pass = corrs["spearman_r"] >= TEST_RETEST_SPEARMAN_TARGET
        mad_pass = corrs["mad"] <= TEST_RETEST_MAD_TARGET
        status = "PASS" if sp_pass and mad_pass else "FAIL"

        rows.append({
            "Category": BUCKET_LABELS[bucket],
            "Pearson r": f"{corrs['pearson_r']:.3f}",
            "Spearman r": f"{corrs['spearman_r']:.3f}",
            "MAD": f"{corrs['mad']:.3f}",
            "n": corrs["n"],
            "Status": status,
        })

    # Overall
    if df.height >= 3:
        r0 = df["run0_median"].to_list()
        r1 = df["run1_median"].to_list()
        corrs = _compute_correlations(r0, r1)
        sp_pass = corrs["spearman_r"] >= TEST_RETEST_SPEARMAN_TARGET
        mad_pass = corrs["mad"] <= TEST_RETEST_MAD_TARGET
        status = "PASS" if sp_pass and mad_pass else "FAIL"
        rows.append({
            "Category": "Overall",
            "Pearson r": f"{corrs['pearson_r']:.3f}",
            "Spearman r": f"{corrs['spearman_r']:.3f}",
            "MAD": f"{corrs['mad']:.3f}",
            "n": corrs["n"],
            "Status": status,
        })

    st.dataframe(pl.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_scatter(df: pl.DataFrame) -> None:
    """Render run 1 vs run 2 scatter colored by category."""
    fig = go.Figure()

    # Diagonal reference
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color="gray", dash="dash"),
    )

    for bucket in BUCKET_ORDER:
        bucket_df = df.filter(pl.col("category") == bucket)
        if bucket_df.is_empty():
            continue

        fig.add_trace(go.Scattergl(
            x=bucket_df["run0_median"].to_list(),
            y=bucket_df["run1_median"].to_list(),
            mode="markers",
            name=BUCKET_LABELS[bucket],
            marker=dict(
                color=CATEGORY_COLORS.get(bucket, "#888888"),
                size=8,
                opacity=0.7,
            ),
            hovertext=bucket_df["proposition"].to_list(),
        ))

    # Annotate with overall correlation
    if df.height >= 3:
        corrs = _compute_correlations(
            df["run0_median"].to_list(), df["run1_median"].to_list()
        )
        fig.add_annotation(
            x=0.05, y=0.95, xref="paper", yref="paper",
            text=f"r={corrs['pearson_r']:.3f}, ρ={corrs['spearman_r']:.3f}",
            showarrow=False, font=dict(size=12),
        )

    fig.update_layout(
        xaxis=dict(title="Run 1 Median Credence", range=[-0.02, 1.02]),
        yaxis=dict(title="Run 2 Median Credence", range=[-0.02, 1.02]),
        height=500,
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_mad_chart(df: pl.DataFrame) -> None:
    """Render MAD bar chart per category."""
    fig = go.Figure()

    for bucket in BUCKET_ORDER:
        bucket_df = df.filter(pl.col("category") == bucket)
        if bucket_df.height < 2:
            continue

        diffs = (
            (bucket_df["run0_median"] - bucket_df["run1_median"]).abs().to_list()
        )
        pt, lo, hi = bootstrap_mean_ci(diffs)

        fig.add_trace(go.Bar(
            x=[BUCKET_LABELS[bucket]],
            y=[pt],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[hi - pt],
                arrayminus=[pt - lo],
            ),
            marker_color=CATEGORY_COLORS.get(bucket, "#888888"),
            showlegend=False,
        ))

    fig.update_layout(
        yaxis=dict(title="Mean Absolute Difference", range=[0, None]),
        xaxis=dict(title="Category"),
        height=350,
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)
