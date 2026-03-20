"""Exp 5: Judge Agreement tab."""

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from credence.viz.constants import (
    AGREEMENT_THRESHOLD,
    CATEGORY_COLORS,
)
from credence.viz.data import load_all_for_agreement
from credence.viz.formatting import short_model
from credence.viz.stats import bootstrap_mean_ci, bootstrap_pass_rate_ci


def render() -> None:
    """Render the judge agreement tab."""
    st.subheader("V4 Judge Agreement: Do different judges produce similar scores?")
    st.caption(
        "Two judges (GPT-5-mini and Claude Sonnet 4.6) independently evaluate every "
        "prompt-response pair, estimating the target model's expressed credence. "
        "We measure how often judges agree within 0.2 across all calibration samples."
    )

    df = load_all_for_agreement()
    if df.is_empty():
        st.info("No judge agreement data. Requires exported data from Exps 1-3.")
        return

    # Filter to samples with both credences present
    working = df.filter(
        pl.col("judge1_credence").is_not_null() & pl.col("judge2_credence").is_not_null()
    )
    if working.is_empty():
        st.warning("No samples with both judge credences available.")
        return

    # Compute |J1 - J2|
    working = working.with_columns(
        (pl.col("judge1_credence") - pl.col("judge2_credence")).abs().alias("abs_diff")
    )
    working = working.with_columns(
        (pl.col("abs_diff") <= AGREEMENT_THRESHOLD).alias("agrees")
    )

    # 1. KPI metrics
    st.subheader("Overall Agreement")
    _render_kpi(working)

    # 2. J1 vs J2 scatter
    st.subheader("Judge 1 vs Judge 2")
    _render_scatter(working)

    # 3. |J1 - J2| histogram
    st.subheader("Absolute Difference Distribution")
    _render_diff_histogram(working)

    # 4. Summary table
    st.subheader("Summary by Target Model")
    _render_summary_table(working)


def _render_kpi(df: pl.DataFrame) -> None:
    """Render headline agreement rate metric."""
    agrees = df["agrees"].to_list()
    pt, lo, hi = bootstrap_pass_rate_ci(agrees)
    n = len(agrees)

    diffs = df["abs_diff"].to_list()
    mean_diff, diff_lo, diff_hi = bootstrap_mean_ci(diffs)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Agreement Rate", f"{pt:.1%}")
        st.caption(f"95% CI: [{lo:.1%}, {hi:.1%}]")
    with col2:
        st.metric("Mean |J1 - J2|", f"{mean_diff:.3f}")
        st.caption(f"95% CI: [{diff_lo:.3f}, {diff_hi:.3f}]")
    with col3:
        st.metric("n samples", f"{n:,}")


def _render_scatter(df: pl.DataFrame) -> None:
    """Render J1 vs J2 scatter with agreement band."""
    fig = go.Figure()

    # Agreement band (within ±0.2 of diagonal)
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color="gray", dash="dash"),
    )
    # Upper and lower threshold lines
    fig.add_shape(
        type="line", x0=0, y0=AGREEMENT_THRESHOLD, x1=1 - AGREEMENT_THRESHOLD, y1=1,
        line=dict(color="lightgray", dash="dot"),
    )
    fig.add_shape(
        type="line", x0=AGREEMENT_THRESHOLD, y0=0, x1=1, y1=1 - AGREEMENT_THRESHOLD,
        line=dict(color="lightgray", dash="dot"),
    )

    # Color by category if available
    if "category" in df.columns:
        categories = sorted(df["category"].unique().to_list())
        for cat in categories:
            cat_df = df.filter(pl.col("category") == cat)
            props = cat_df["proposition"].to_list() if "proposition" in cat_df.columns else []
            hover = [p[:60] + "..." if len(p) > 60 else p for p in props] if props else None
            fig.add_trace(go.Scatter(
                x=cat_df["judge1_credence"].to_list(),
                y=cat_df["judge2_credence"].to_list(),
                mode="markers",
                name=cat,
                marker=dict(
                    color=CATEGORY_COLORS.get(cat, "#888888"),
                    opacity=0.6,
                    size=6,
                ),
                hovertext=hover,
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df["judge1_credence"].to_list(),
            y=df["judge2_credence"].to_list(),
            mode="markers",
            marker=dict(opacity=0.6, size=6),
        ))

    fig.update_layout(
        xaxis=dict(title="Judge 1 Credence", range=[-0.02, 1.02]),
        yaxis=dict(title="Judge 2 Credence", range=[-0.02, 1.02]),
        height=500,
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_diff_histogram(df: pl.DataFrame) -> None:
    """Render histogram of |J1 - J2| with threshold line."""
    diffs = df["abs_diff"].to_list()
    mean_diff = sum(diffs) / len(diffs)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=diffs,
        nbinsx=30,
        marker_color="#636EFA",
        opacity=0.7,
    ))

    fig.add_vline(
        x=AGREEMENT_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({AGREEMENT_THRESHOLD})",
    )
    fig.add_vline(
        x=mean_diff,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Mean ({mean_diff:.3f})",
    )

    fig.update_layout(
        xaxis=dict(title="|Judge 1 - Judge 2|", range=[0, 1]),
        yaxis=dict(title="Count"),
        height=350,
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_summary_table(df: pl.DataFrame) -> None:
    """Render per-model agreement summary."""
    rows = []
    for model in sorted(df["target_model"].unique().to_list()):
        model_df = df.filter(pl.col("target_model") == model)
        agrees = model_df["agrees"].to_list()
        diffs = model_df["abs_diff"].to_list()
        pt, lo, hi = bootstrap_pass_rate_ci(agrees)
        mean_diff, _, _ = bootstrap_mean_ci(diffs)
        rows.append({
            "Model": short_model(model),
            "Agreement Rate": f"{pt:.1%} [{lo:.1%}, {hi:.1%}]",
            "Mean |Diff|": f"{mean_diff:.3f}",
            "n": len(agrees),
        })

    st.dataframe(pl.DataFrame(rows), use_container_width=True, hide_index=True)
