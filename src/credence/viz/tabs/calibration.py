"""Exp 1: Five-Bucket Calibration tab."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import streamlit as st

from credence.viz.constants import (
    BUCKET_LABELS,
    BUCKET_ORDER,
    BUCKET_RANGES,
    CATEGORY_COLORS,
)
from credence.viz.data import load_calibration
from credence.viz.formatting import provider_color, short_model
from credence.viz.stats import in_bucket


def render() -> None:
    """Render the calibration tab."""
    st.subheader("V1 Calibration: Do credences match expected ranges?")
    st.caption(
        "We construct five sets of 20 propositions with target credence ranges "
        "(unequivocally false, likely false, uncertain, likely true, unequivocally true) "
        "and check whether median credences fall within the expected bucket. "
        "32 prompts per proposition, 2 independent runs."
    )

    df = load_calibration()
    if df.is_empty():
        st.info("No calibration data. Run validation to generate `.data/calibration/results.jsonl`.")
        return

    # Determine available runs
    available_runs = sorted(df["run_id"].unique().to_list()) if "run_id" in df.columns else []
    has_multiple_runs = len(available_runs) > 1

    # Controls
    cols = st.columns(2 if has_multiple_runs else 1)
    with cols[0]:
        agg_level = st.radio(
            "Aggregation",
            ["Proposition (median)", "Prompt"],
            horizontal=True,
            key="cal_agg",
        )
    if has_multiple_runs:
        with cols[1]:
            run_choice = st.radio(
                "Run",
                ["All runs"] + [f"Run {r}" for r in available_runs],
                horizontal=True,
                key="cal_run",
            )
            if run_choice != "All runs":
                selected_run = int(run_choice.split()[-1])
                df = df.filter(pl.col("run_id") == selected_run)

    credence_col = "consensus_credence"

    # Filter to valid credences
    working = df.filter(pl.col(credence_col).is_not_null())
    if working.is_empty():
        st.warning("No valid credence data for the selected judge mode.")
        return

    target_models = sorted(working["target_model"].unique().to_list())
    use_proposition_level = agg_level.startswith("Proposition")

    # 1. Distribution box plots with % in target annotations
    st.subheader("Credence Distributions")
    _render_distribution_plots(working, target_models, credence_col, use_proposition_level)

    # 2. Judge & consensus rates by bucket
    st.subheader("Judge & Consensus Rates")
    _render_judge_stats(df)

    # 3. Per-proposition detail
    with st.expander("Per-Proposition Detail"):
        _render_proposition_detail(working, credence_col)


def _render_judge_stats(df: pl.DataFrame) -> None:
    """Render donut charts: consensus/uninformative/disagreement per bucket."""
    if "judge1_informative" not in df.columns:
        return

    # Collect per-bucket stats
    bucket_stats = []
    for bucket in BUCKET_ORDER:
        bdf = df.filter(pl.col("category") == bucket)
        n = bdf.height
        if n == 0:
            continue

        uninf = bdf.filter(
            ~pl.col("judge1_informative") | ~pl.col("judge2_informative")
        ).height
        consensus = bdf.filter(pl.col("consensus_credence").is_not_null()).height
        disagree = n - consensus - uninf

        bucket_stats.append((BUCKET_LABELS[bucket], consensus, uninf, disagree))

    if not bucket_stats:
        return

    n_buckets = len(bucket_stats)
    fig = make_subplots(
        rows=1, cols=n_buckets,
        specs=[[{"type": "pie"}] * n_buckets],
        subplot_titles=[s[0] for s in bucket_stats],
    )

    for i, (_, consensus, uninf, disagree) in enumerate(bucket_stats):
        fig.add_trace(go.Pie(
            labels=["Consensus", "Uninformative", "Disagreement"],
            values=[consensus, uninf, disagree],
            marker=dict(colors=["#2ecc71", "#e74c3c", "#f39c12"]),
            textinfo="percent",
            textposition="inside",
            hole=0.4,
        ), row=1, col=i + 1)

    fig.update_layout(
        height=250, margin=dict(t=40, b=20), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)


def _prop_group_cols(df: pl.DataFrame) -> list[str]:
    """Group-by columns for proposition-level aggregation.

    Includes run_id when present so each run is a separate observation.
    """
    if "run_id" in df.columns:
        return ["proposition", "run_id"]
    return ["proposition"]


def _render_distribution_plots(
    df: pl.DataFrame,
    target_models: list[str],
    credence_col: str,
    proposition_level: bool,
) -> None:
    """Render box plots with expected-range shading and % in target annotations."""
    fig = go.Figure()

    # Shaded expected-range rectangles
    for i, bucket in enumerate(BUCKET_ORDER):
        lo, hi = BUCKET_RANGES[bucket]
        fig.add_shape(
            type="rect",
            x0=i - 0.4,
            x1=i + 0.4,
            y0=lo,
            y1=hi,
            fillcolor=CATEGORY_COLORS.get(bucket, "#2ecc71"),
            opacity=0.12,
            line_width=0,
        )

    for model in target_models:
        x_positions = []
        y_values = []

        for i, bucket in enumerate(BUCKET_ORDER):
            model_bucket = df.filter(
                (pl.col("target_model") == model) & (pl.col("category") == bucket)
            )
            if model_bucket.is_empty():
                continue

            if proposition_level:
                vals = (
                    model_bucket.group_by(_prop_group_cols(model_bucket))
                    .agg(pl.col(credence_col).median().alias("val"))
                    .filter(pl.col("val").is_not_null())
                    ["val"]
                    .to_list()
                )
            else:
                vals = model_bucket.filter(
                    pl.col(credence_col).is_not_null()
                )[credence_col].to_list()

            x_positions.extend([BUCKET_LABELS[bucket]] * len(vals))
            y_values.extend(vals)

            # Compute % in target and annotate
            if vals:
                n_in = sum(1 for v in vals if in_bucket(v, bucket))
                pct = n_in / len(vals)
                fig.add_annotation(
                    x=BUCKET_LABELS[bucket],
                    y=1.05,
                    text=f"{pct:.0%}",
                    showarrow=False,
                    font=dict(size=12),
                )

        fig.add_trace(go.Box(
            x=x_positions,
            y=y_values,
            name=short_model(model),
            marker=dict(color=provider_color(model), opacity=0.4, size=5),
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
        ))

    fig.update_layout(
        yaxis=dict(title="Credence", range=[-0.02, 1.12]),
        boxmode="group",
        height=450,
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)



def _render_proposition_detail(
    df: pl.DataFrame,
    credence_col: str,
) -> None:
    """Render per-proposition breakdown table."""
    detail = (
        df.filter(pl.col(credence_col).is_not_null())
        .group_by([*_prop_group_cols(df), "category", "target_model"])
        .agg([
            pl.col(credence_col).median().alias("median_credence"),
            pl.col(credence_col).count().alias("n_samples"),
        ])
        .sort(["category", "proposition", "target_model"])
    )

    if detail.is_empty():
        st.write("No data.")
        return

    # Add pass/fail column
    detail = detail.with_columns(
        pl.struct(["median_credence", "category"])
        .map_elements(
            lambda row: "PASS" if row["median_credence"] is not None and in_bucket(row["median_credence"], row["category"]) else "FAIL",
            return_dtype=pl.Utf8,
        )
        .alias("status")
    )

    st.dataframe(detail, use_container_width=True, hide_index=True)
