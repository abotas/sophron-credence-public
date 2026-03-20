"""Exp 2: Negation Consistency tab."""

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from credence.viz.data import load_negation
from credence.viz.formatting import truncate
from credence.viz.stats import bootstrap_mean_ci


_UNEQUIVOCALLY = ["unequivocally_true", "unequivocally_false"]
_LIKELY = ["likely_true", "likely_false"]

_COLOR_UNEQUIV = "#e74c3c"  # red
_COLOR_LIKELY = "#3498db"   # blue


def _compute_consensus_stats(df: pl.DataFrame) -> tuple[int, int, int]:
    """Compute consensus/uninformative/disagreement counts for combined P and not-P sides."""
    n = len(df)
    total_uninf = total_disagree = total_ok = 0
    for side in ["p", "notp"]:
        j1_col = f"{side}_judge1_credence"
        j2_col = f"{side}_judge2_credence"
        cons_col = f"{side}_consensus"

        uninf = df.filter(pl.col(j1_col).is_null() | pl.col(j2_col).is_null()).height
        disagree = df.filter(
            pl.col(j1_col).is_not_null() & pl.col(j2_col).is_not_null() & pl.col(cons_col).is_null()
        ).height
        total_uninf += uninf
        total_disagree += disagree
        total_ok += n - uninf - disagree
    return total_ok, total_uninf, total_disagree


def _render_judge_stats(df: pl.DataFrame) -> None:
    """Render donut charts showing consensus breakdown for Unequivocally and Likely."""
    if len(df) == 0 or "p_judge1_credence" not in df.columns:
        return

    from plotly.subplots import make_subplots

    groups = [
        ("Unequivocal", df.filter(pl.col("category").is_in(_UNEQUIVOCALLY)) if "category" in df.columns else df.head(0)),
        ("Likely", df.filter(pl.col("category").is_in(_LIKELY)) if "category" in df.columns else df.head(0)),
    ]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=[g[0] for g in groups],
    )

    for i, (_, sub_df) in enumerate(groups):
        if sub_df.is_empty():
            continue
        ok, uninf, disagree = _compute_consensus_stats(sub_df)
        fig.add_trace(go.Pie(
            labels=["Consensus", "Uninformative", "Disagreement"],
            values=[ok, uninf, disagree],
            marker=dict(colors=["#2ecc71", "#e74c3c", "#f39c12"]),
            textinfo="percent",
            textposition="inside",
            hole=0.4,
        ), row=1, col=i + 1)

    fig.update_layout(
        height=250, margin=dict(t=40, b=20), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.subheader("Judge & Consensus Rates")
    st.plotly_chart(fig, use_container_width=True)


def _proposition_level_errors(df: pl.DataFrame) -> pl.DataFrame:
    """Compute proposition-level negation errors per the preregistration.

    For each pair: median(consensus_P), median(consensus_notP),
    then |median_P + median_notP - 1|.
    """
    return (
        df.filter(
            pl.col("p_consensus").is_not_null() & pl.col("notp_consensus").is_not_null()
        )
        .group_by("pair_id")
        .agg([
            pl.col("p_consensus").median().alias("median_p"),
            pl.col("notp_consensus").median().alias("median_notp"),
            pl.col("category").first().alias("category"),
            pl.col("proposition_p").first().alias("proposition_p"),
            pl.col("proposition_not_p").first().alias("proposition_not_p"),
        ])
        .with_columns(
            (pl.col("median_p") + pl.col("median_notp") - 1.0).abs().alias("consistency_error")
        )
    )


def render() -> None:
    """Render the negation consistency tab."""
    df = load_negation()
    if df.is_empty():
        st.info("No negation data. Export results to `negation.jsonl`.")
        return

    # --- Filters ---
    level = st.radio("Aggregation", ["Proposition (median)", "Prompt"], horizontal=True)

    # Filter to samples with consistency_error computed
    working = df.filter(pl.col("consistency_error").is_not_null())
    if working.is_empty():
        st.warning("No samples with computed consistency errors.")
        return

    # 1. Error distribution + signed error side by side
    left, right = st.columns(2)
    with left:
        st.subheader("Absolute Error Distribution")
        _render_abs_error_histogram(working, level)
    with right:
        st.subheader("Signed Error Distribution")
        _render_signed_error_histogram(working, level)

    # 2. Judge & consensus rates
    _render_judge_stats(df)

    # 3. Negation pair detail
    st.subheader("Negation Pair Detail")
    _render_pair_detail(working)



def _get_errors_by_group(df: pl.DataFrame, level: str) -> tuple[list, list, list]:
    """Get absolute errors split by category group. Returns (all, unequiv, likely)."""
    prop_level = level.startswith("Proposition") and "pair_id" in df.columns

    def _errors(sub_df):
        if prop_level:
            return _proposition_level_errors(sub_df)["consistency_error"].to_list()
        return sub_df["consistency_error"].drop_nulls().to_list()

    all_errors = _errors(df)
    unequiv_df = df.filter(pl.col("category").is_in(_UNEQUIVOCALLY)) if "category" in df.columns else df.head(0)
    likely_df = df.filter(pl.col("category").is_in(_LIKELY)) if "category" in df.columns else df.head(0)
    return all_errors, _errors(unequiv_df), _errors(likely_df)


def _stats_line(errors: list, label: str) -> str:
    """Format a stats line for a group of errors."""
    if not errors:
        return ""
    pt, ci_lo, ci_hi = bootstrap_mean_ci(errors)
    sorted_e = sorted(errors)
    median_e = sorted_e[len(sorted_e) // 2]
    return f"**{label}**: mean=`{pt:.3f}` [{ci_lo:.3f}, {ci_hi:.3f}], median=`{median_e:.3f}`, n={len(errors)}"


def _render_abs_error_histogram(df: pl.DataFrame, level: str) -> None:
    """Render histogram of absolute consistency errors colored by category group."""
    all_errors, unequiv_errors, likely_errors = _get_errors_by_group(df, level)

    st.markdown(_stats_line(all_errors, "Overall"))
    st.markdown(_stats_line(unequiv_errors, "Unequivocal"))
    st.markdown(_stats_line(likely_errors, "Likely"))

    fig = go.Figure()
    abs_bins = dict(start=0, end=1, size=0.025)
    if unequiv_errors:
        fig.add_trace(go.Histogram(
            x=unequiv_errors,
            name="Unequivocal",
            marker=dict(color=_COLOR_UNEQUIV),
            opacity=0.7,
            xbins=abs_bins,
        ))
    if likely_errors:
        fig.add_trace(go.Histogram(
            x=likely_errors,
            name="Likely",
            marker=dict(color=_COLOR_LIKELY),
            opacity=0.7,
            xbins=abs_bins,
        ))

    fig.update_layout(
        xaxis=dict(title="Absolute Error |J(P) + J(¬P) − 1|"),
        yaxis=dict(title="Count"),
        barmode="overlay",
        height=400,
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Absolute error = |J(P) + J(¬P) − 1|. Perfect negation consistency → 0.")


def _get_signed_errors_by_group(df: pl.DataFrame, level: str) -> tuple[list, list, list]:
    """Get signed errors split by category group. Returns (all, unequiv, likely)."""
    working = df.filter(
        pl.col("p_consensus").is_not_null() & pl.col("notp_consensus").is_not_null()
    )
    if working.is_empty():
        return [], [], []

    prop_level = level.startswith("Proposition") and "pair_id" in working.columns

    def _signed(sub_df):
        if sub_df.is_empty():
            return []
        if prop_level:
            agg = sub_df.group_by("pair_id").agg([
                pl.col("p_consensus").median(),
                pl.col("notp_consensus").median(),
            ])
            return [(p + np - 1) for p, np in zip(
                agg["p_consensus"].to_list(), agg["notp_consensus"].to_list()
            )]
        return [(p + np - 1) for p, np in zip(
            sub_df["p_consensus"].to_list(), sub_df["notp_consensus"].to_list()
        )]

    all_signed = _signed(working)
    unequiv_signed = _signed(working.filter(pl.col("category").is_in(_UNEQUIVOCALLY))) if "category" in working.columns else []
    likely_signed = _signed(working.filter(pl.col("category").is_in(_LIKELY))) if "category" in working.columns else []
    return all_signed, unequiv_signed, likely_signed


def _signed_stats_line(errors: list, label: str) -> str:
    """Format a stats line for signed errors."""
    if not errors:
        return ""
    mean_e = sum(errors) / len(errors)
    sorted_e = sorted(errors)
    median_e = sorted_e[len(sorted_e) // 2]
    return f"**{label}**: mean=`{mean_e:.3f}`, median=`{median_e:.3f}`, n={len(errors)}"


def _render_signed_error_histogram(df: pl.DataFrame, level: str) -> None:
    """Histogram of signed error J(P) + J(¬P) − 1 colored by category group."""
    all_signed, unequiv_signed, likely_signed = _get_signed_errors_by_group(df, level)

    if not all_signed:
        st.write("No P/not-P consensus columns available.")
        return

    st.markdown(_signed_stats_line(all_signed, "Overall"))
    st.markdown(_signed_stats_line(unequiv_signed, "Unequivocal"))
    st.markdown(_signed_stats_line(likely_signed, "Likely"))

    fig = go.Figure()
    signed_bins = dict(start=-1, end=1, size=0.025)
    if unequiv_signed:
        fig.add_trace(go.Histogram(
            x=unequiv_signed,
            name="Unequivocal",
            marker=dict(color=_COLOR_UNEQUIV),
            opacity=0.7,
            xbins=signed_bins,
        ))
    if likely_signed:
        fig.add_trace(go.Histogram(
            x=likely_signed,
            name="Likely",
            marker=dict(color=_COLOR_LIKELY),
            opacity=0.7,
            xbins=signed_bins,
        ))

    fig.add_vline(x=0, line=dict(color="gray", dash="dash"))
    fig.update_layout(
        xaxis=dict(title="Signed error: J(P) + J(¬P) − 1"),
        yaxis=dict(title="Count"),
        barmode="overlay",
        height=400,
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Signed error = J(P) + J(¬P) − 1. Perfect negation consistency → 0.")


def _render_pair_detail(df: pl.DataFrame) -> None:
    """Render stacked horizontal bars showing P + ¬P median credences per pair."""
    pair_df = _proposition_level_errors(df)
    if pair_df.is_empty():
        return

    pair_df = pair_df.with_columns(
        (pl.col("median_p") + pl.col("median_notp")).alias("credence_sum")
    ).sort("credence_sum")

    props_p = pair_df["proposition_p"].to_list()
    props_notp = pair_df["proposition_not_p"].to_list()
    categories = pair_df["category"].to_list()
    median_p = pair_df["median_p"].to_list()
    median_notp = pair_df["median_notp"].to_list()
    errors = pair_df["consistency_error"].to_list()
    sums = pair_df["credence_sum"].to_list()

    labels = []
    for p, cat in zip(props_p, categories):
        tag = "U" if "unequiv" in cat else "L"
        short = truncate(p, 45)
        labels.append(f"[{tag}] {short}")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels,
        x=median_p,
        orientation="h",
        name="P (median)",
        marker=dict(color="#3498db"),
        customdata=list(zip(props_p, median_p, median_notp, sums, errors)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "P: %{customdata[1]:.3f}<br>"
            "¬P: %{customdata[2]:.3f}<br>"
            "Sum: %{customdata[3]:.3f}<br>"
            "Error: %{customdata[4]:.3f}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Bar(
        y=labels,
        x=median_notp,
        orientation="h",
        name="¬P (median)",
        marker=dict(color="#e74c3c"),
        customdata=list(zip(props_notp, median_p, median_notp, sums, errors)),
        hovertemplate=(
            "<b>¬: %{customdata[0]}</b><br>"
            "P: %{customdata[1]:.3f}<br>"
            "¬P: %{customdata[2]:.3f}<br>"
            "Sum: %{customdata[3]:.3f}<br>"
            "Error: %{customdata[4]:.3f}<extra></extra>"
        ),
    ))

    fig.add_vline(x=1.0, line=dict(color="gray", dash="dash", width=2))

    max_sum = max(sums) if sums else 1.5
    n_pairs = len(labels)
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Credence Sum (P + ¬P)", range=[0, max(1.5, max_sum + 0.1)]),
        height=max(300, 30 * n_pairs),
        margin=dict(t=30, b=20, l=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each bar shows median P credence (blue) + median ¬P credence (red). "
        "Perfect negation consistency → bar reaches exactly 1.0 (dashed line). "
        "[U] = Unequivocal, [L] = Likely."
    )
