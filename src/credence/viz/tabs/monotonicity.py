"""Exp 3: Monotonicity tab."""

import statistics

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from credence.viz.data import load_monotonicity
from credence.viz.formatting import natural_sort_key, provider_color, truncate
from credence.viz.stats import bootstrap_pass_rate_ci


def _render_judge_stats(df: pl.DataFrame) -> None:
    """Render donut charts showing consensus breakdown per proposition position."""
    if "judgments" not in df.columns:
        return

    from plotly.subplots import make_subplots

    # Determine max number of positions
    max_pos = max(
        (len(j) for judgments in df["judgments"].to_list() if judgments for j in [judgments]),
        default=0,
    )
    if max_pos == 0:
        return

    # Count per position
    per_pos: list[dict[str, int]] = [{"ok": 0, "uninf": 0, "disagree": 0} for _ in range(max_pos)]
    for judgments, creds in zip(df["judgments"].to_list(), df["credences"].to_list()):
        if not judgments:
            continue
        for pi, j in enumerate(judgments):
            if pi >= max_pos:
                break
            has_consensus = creds and pi < len(creds) and creds[pi] is not None
            j1_inf = j.get("j1", {}).get("informative", True)
            j2_inf = j.get("j2", {}).get("informative", True)
            if has_consensus:
                per_pos[pi]["ok"] += 1
            elif not j1_inf or not j2_inf:
                per_pos[pi]["uninf"] += 1
            else:
                per_pos[pi]["disagree"] += 1

    labels = ["Consensus", "Uninformative", "Disagreement"]
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    position_labels = ["Broadest (elicited)", "Middle", "Narrowest"]

    fig = make_subplots(
        rows=1, cols=max_pos,
        specs=[[{"type": "pie"}] * max_pos],
        subplot_titles=[position_labels[i] if i < len(position_labels) else f"Position {i+1}" for i in range(max_pos)],
    )
    for i, counts in enumerate(per_pos):
        fig.add_trace(go.Pie(
            labels=labels,
            values=[counts["ok"], counts["uninf"], counts["disagree"]],
            marker=dict(colors=colors),
            textinfo="percent+label",
            hole=0.4,
            showlegend=False,
        ), row=1, col=i + 1)

    fig.update_layout(height=300, margin=dict(t=40, b=20))
    st.subheader("Judge & Consensus Rates")
    st.plotly_chart(fig, width="stretch")


def _series_proposition_level_pass(df: pl.DataFrame, series_id: str) -> bool | None:
    """Check if a series passes monotonicity at the proposition level.

    Computes median credence per proposition position, then checks
    if medians are non-increasing. Returns None if any position has
    no valid credences.
    """
    series_df = df.filter(pl.col("series_id") == series_id)
    if series_df.is_empty():
        return None

    first_row = series_df.row(0, named=True)
    propositions = first_row.get("propositions", [])
    if not propositions:
        return None

    n_props = len(propositions)
    per_prop_credences: list[list[float]] = [[] for _ in range(n_props)]
    for row in series_df.iter_rows(named=True):
        credences = row["credences"]
        if credences is None:
            continue
        for i, c in enumerate(credences):
            if i < n_props and c is not None:
                per_prop_credences[i].append(c)

    medians = []
    for creds in per_prop_credences:
        if not creds:
            return None
        medians.append(statistics.median(creds))

    return all(medians[i] >= medians[i + 1] for i in range(len(medians) - 1))


def render() -> None:
    """Render the monotonicity tab."""
    df = load_monotonicity()
    if df.is_empty():
        st.info("No monotonicity data. Export results to `monotonicity.jsonl`.")
        return

    # Filter to samples with is_monotonic computed
    working = df.filter(pl.col("is_monotonic").is_not_null())
    if working.is_empty():
        st.warning("No samples with computed monotonicity status.")
        return

    # --- Headline metrics ---
    series_ids = sorted(working["series_id"].unique().to_list(), key=natural_sort_key)

    # Proposition-level: do median credences per series respect ordering?
    # Precompute once and pass to _render_single_series to avoid duplicate work
    prop_pass_by_series: dict[str, bool | None] = {}
    prop_passes = []
    for sid in series_ids:
        result = _series_proposition_level_pass(working, sid)
        prop_pass_by_series[sid] = result
        if result is not None:
            prop_passes.append(result)

    # Sample-level: fraction of individual samples with zero violations
    sample_passes = working["is_monotonic"].to_list()

    if prop_passes:
        prop_pt, prop_lo, prop_hi = bootstrap_pass_rate_ci(prop_passes)
        st.markdown(
            f"**Median proposition credence monotonicity**: `{prop_pt:.0%}` [{prop_lo:.0%}, {prop_hi:.0%}] "
            f"({sum(prop_passes)}/{len(prop_passes)} series)"
        )
    if sample_passes:
        sample_pt, sample_lo, sample_hi = bootstrap_pass_rate_ci(sample_passes)
        st.markdown(
            f"**Prompt credence monotonicity**: `{sample_pt:.0%}` [{sample_lo:.0%}, {sample_hi:.0%}] "
            f"({sum(sample_passes)}/{len(sample_passes)} samples)"
        )

    st.caption(
        "Median proposition credence: median credences per proposition in each series must be non-increasing. "
        "Prompt credence: fraction of individual prompt-response pairs with zero ordering violations."
    )

    # 1. Judge & consensus rates
    _render_judge_stats(df)

    # 2. Series box plots with median lines
    st.subheader("Series Detail")
    _render_series_box_plots(working, prop_pass_by_series)



def _render_series_box_plots(df: pl.DataFrame, prop_pass_by_series: dict[str, bool | None]) -> None:
    """Render horizontal box plots per proposition in each series, with median connecting lines."""
    if "series_id" not in df.columns or "credences" not in df.columns:
        st.write("Missing series_id or credences columns.")
        return

    series_ids = sorted(df["series_id"].unique().to_list(), key=natural_sort_key)
    if not series_ids:
        return

    # Render in 2-column grid
    for row_start in range(0, len(series_ids), 2):
        cols = st.columns(2)
        for col_idx in range(2):
            idx = row_start + col_idx
            if idx >= len(series_ids):
                break
            with cols[col_idx]:
                _render_single_series(df, series_ids[idx], prop_pass_by_series.get(series_ids[idx]))


def _render_single_series(df: pl.DataFrame, series_id, prop_pass: bool | None = None) -> None:
    """Render a single monotonicity series box plot."""
    series_df = df.filter(pl.col("series_id") == series_id)
    if series_df.is_empty():
        return

    first_row = series_df.row(0, named=True)
    propositions = first_row.get("propositions", [])
    if not propositions:
        return

    n_props = len(propositions)
    prop_labels = [truncate(p, 50) for p in propositions]

    # Collect per-proposition credences across all samples in this series
    # all_credences: every row with consensus at that position (for box plots)
    # full_consensus_credences: only rows with consensus at ALL positions (for median line)
    per_prop_credences: list[list[float]] = [[] for _ in range(n_props)]
    full_consensus_credences: list[list[float]] = [[] for _ in range(n_props)]
    for row in series_df.iter_rows(named=True):
        credences = row["credences"]
        if credences is None:
            continue
        # Check if all positions have consensus
        has_full = (
            len(credences) >= n_props
            and all(credences[i] is not None for i in range(n_props))
        )
        for i, c in enumerate(credences):
            if i < n_props and c is not None:
                per_prop_credences[i].append(c)
                if has_full:
                    full_consensus_credences[i].append(c)

    target_models = sorted(series_df["target_model"].unique().to_list())
    color = provider_color(target_models[0]) if target_models else "#636EFA"

    # Pass/fail info
    prop_status = "PASS" if prop_pass else "FAIL" if prop_pass is not None else "N/A"
    passes = series_df["is_monotonic"].to_list()
    sample_rate = sum(passes) / len(passes) if passes else 0
    n_full = len(full_consensus_credences[0]) if full_consensus_credences else 0

    st.markdown(
        f"**Series {series_id}** — Median prop: {prop_status} | "
        f"Prompt: {sample_rate:.0%} ({sum(passes)}/{len(passes)}) | "
        f"Full consensus: {n_full}/{len(series_df)}"
    )

    fig = go.Figure()

    # Medians from rows with full consensus only (matches pass/fail logic)
    full_medians = []
    for i in range(n_props):
        creds = full_consensus_credences[i]
        full_medians.append(statistics.median(creds) if creds else None)

    medians = []
    for i in range(n_props):
        creds = per_prop_credences[i]
        med = full_medians[i]
        med_str = f" med:{med:.2f}" if med is not None else ""
        label = f"{prop_labels[i]}{med_str}"
        full_prop = propositions[i]
        medians.append((med, label))
        fig.add_trace(go.Box(
            x=creds,
            y=[label] * len(creds),
            orientation="h",
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            marker=dict(opacity=0.4, size=4, color=color),
            line=dict(color=color),
            showlegend=False,
            hovertext=[full_prop] * len(creds),
            hoverinfo="text+x",
        ))

    # Connecting line between medians
    valid_points = [(m, lbl) for m, lbl in medians if m is not None]
    if len(valid_points) > 1:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in valid_points],
            y=[p[1] for p in valid_points],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=10, symbol="diamond", color=color),
            showlegend=False,
        ))

    fig.update_layout(
        xaxis=dict(range=[0, 1], title="Credence"),
        height=60 + 70 * n_props,
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig, width="stretch")


