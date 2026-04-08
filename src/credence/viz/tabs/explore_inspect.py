"""Exploration Inspect tab — Deep-dive into individual exploration samples.

Flow: Select proposition → select prompt → view all model responses + judgements.
Prompts are shared across models, so we group by prompt_text and show per-model
responses side by side.
"""

import statistics

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from scipy.stats import linregress

from credence.viz.data import load_exploration
from credence.viz.formatting import provider_color, short_model, truncate, unslugify


def render() -> None:
    """Render the exploration inspect tab."""
    st.subheader("Inspect: Explore prompts and model responses")
    st.caption(
        "Select a proposition, then pick a prompt to see how each model responded "
        "and how the judges scored each response."
    )

    df = load_exploration()
    if df.is_empty():
        st.info("No exploration data. Run `uvr python run_exploration.py` first.")
        return

    if "prompt_text" not in df.columns or "response_text" not in df.columns:
        st.warning("Re-export with the latest exporter to populate prompt/response text.")
        return

    # Domain filter
    domains = sorted(df["domain"].unique().to_list())
    if len(domains) > 1:
        selected_domain = st.selectbox(
            "Domain", ["All"] + domains,
            format_func=lambda d: unslugify(d) if d != "All" else d,
            key="explore_inspect_domain",
        )
        if selected_domain != "All":
            df = df.filter(pl.col("domain") == selected_domain)

    # ── Step 1: Select proposition ─────────────────────────────────────────
    sort_by = st.radio(
        "Sort propositions by",
        ["Median credence", "Sycophancy slope"],
        horizontal=True,
        key="explore_inspect_sort",
    )

    prop_stats = (
        df.group_by("proposition")
        .agg([
            pl.col("consensus_credence").drop_nulls().median().alias("median"),
            pl.len().alias("n_total"),
        ])
    )

    if prop_stats.is_empty():
        st.warning("No propositions found.")
        return

    # Compute per-proposition slope if needed
    has_valence = "consensus_author_valence" in df.columns
    if has_valence:
        slopes = {}
        for prop in prop_stats["proposition"].to_list():
            sub = df.filter(
                (pl.col("proposition") == prop)
                & pl.col("consensus_credence").is_not_null()
                & pl.col("consensus_author_valence").is_not_null()
            )
            if sub.height >= 5:
                x = sub["consensus_author_valence"].to_numpy()
                y = sub["consensus_credence"].to_numpy()
                reg = linregress(x, y)
                slopes[prop] = reg.slope
            else:
                slopes[prop] = None
        prop_stats = prop_stats.with_columns(
            pl.col("proposition").map_elements(lambda p: slopes.get(p), return_dtype=pl.Float64).alias("slope")
        )

    if sort_by == "Sycophancy slope" and has_valence:
        prop_stats = prop_stats.sort(["slope", "proposition"], descending=[True, False], nulls_last=True)
    else:
        prop_stats = prop_stats.sort(["median", "proposition"], descending=[True, False])

    props = prop_stats["proposition"].to_list()
    medians = {r["proposition"]: r["median"] for r in prop_stats.to_dicts()}
    slope_vals = {r["proposition"]: r.get("slope") for r in prop_stats.to_dicts()} if has_valence else {}

    prop_to_label = {}
    for p in props:
        m = medians[p]
        m_str = f"{m:.2f}" if m is not None else "N/A"
        if sort_by == "Sycophancy slope" and has_valence:
            s = slope_vals.get(p)
            s_str = f"{s:+.3f}" if s is not None else "N/A"
            prop_to_label[p] = f"slope {s_str}  med {m_str} — {truncate(p, 50)}"
        else:
            prop_to_label[p] = f"{m_str} — {truncate(p, 60)}"

    selected_prop = st.selectbox(
        "Proposition",
        props,
        format_func=lambda p: prop_to_label[p],
        key="explore_inspect_prop",
    )

    prop_df = df.filter(pl.col("proposition") == selected_prop)

    # Header with overall stats
    consensus_vals = prop_df.filter(
        pl.col("consensus_credence").is_not_null()
    )["consensus_credence"].to_list()
    median_val = statistics.median(consensus_vals) if consensus_vals else None
    m_str = f"{median_val:.2f}" if median_val is not None else "N/A"
    st.markdown(f"**{selected_prop}**  \nMedian credence: `{m_str}`")

    # Histogram + donut
    _render_distribution_and_donut(prop_df)

    # Valence vs credence scatter
    if "consensus_author_valence" in prop_df.columns:
        _render_valence_scatter(prop_df)

    st.divider()

    # ── Step 2: Select prompt ──────────────────────────────────────────────
    _render_prompt_browser(prop_df)


def _render_distribution_and_donut(prop_df: pl.DataFrame) -> None:
    """Render histogram + donut chart."""
    left, right = st.columns([3, 2])

    with left:
        consensus_vals = prop_df.filter(
            pl.col("consensus_credence").is_not_null()
        )["consensus_credence"].to_list()

        if not consensus_vals:
            st.write("No consensus credences.")
            return

        models = sorted(prop_df["target_model"].unique().to_list()) if "target_model" in prop_df.columns else []
        fig = go.Figure()

        if len(models) > 1:
            for model in models:
                model_vals = prop_df.filter(
                    (pl.col("target_model") == model)
                    & pl.col("consensus_credence").is_not_null()
                )["consensus_credence"].to_list()
                fig.add_trace(go.Histogram(
                    x=model_vals,
                    name=short_model(model),
                    marker_color=provider_color(model),
                    opacity=0.7,
                    xbins=dict(start=0, end=1, size=0.05),
                ))
            fig.update_layout(barmode="overlay")
        else:
            fig.add_trace(go.Histogram(
                x=consensus_vals, marker_color="#636EFA", opacity=0.7,
                xbins=dict(start=0, end=1, size=0.05),
            ))

        fig.update_layout(
            xaxis=dict(title="Consensus Credence", range=[0, 1]),
            yaxis=dict(title="Count"),
            height=280, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        n = prop_df.height
        consensus = prop_df.filter(pl.col("consensus_credence").is_not_null()).height
        uninf = 0
        if "judge1_informative" in prop_df.columns:
            uninf = prop_df.filter(
                ~pl.col("judge1_informative") | ~pl.col("judge2_informative")
            ).height
        disagree = n - consensus - uninf

        labels = ["Consensus", "Uninformative", "Disagreement"]
        colors = ["#2ecc71", "#e74c3c", "#f39c12"]
        fig = go.Figure(go.Pie(
            labels=labels, values=[consensus, uninf, disagree],
            marker=dict(colors=colors),
            textinfo="percent+label", textposition="inside",
            hole=0.4,
        ))
        fig.update_layout(height=280, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def _render_valence_scatter(prop_df: pl.DataFrame) -> None:
    """Scatter of user valence vs credence with per-model OLS lines for this proposition."""
    valid = prop_df.filter(
        pl.col("consensus_credence").is_not_null()
        & pl.col("consensus_author_valence").is_not_null()
    )
    if valid.height < 5:
        return

    use_logit = st.checkbox("Logit scale", value=False, key="explore_inspect_logit")

    def _to_logit(arr: np.ndarray) -> np.ndarray:
        clipped = np.clip(arr, 0.01, 0.99)
        return np.log(clipped / (1 - clipped))

    models = sorted(valid["target_model"].unique().to_list())
    fig = go.Figure()

    for model_id in models:
        model_data = valid.filter(pl.col("target_model") == model_id)
        if model_data.height < 3:
            continue
        x = model_data["consensus_author_valence"].to_numpy()
        y_raw = model_data["consensus_credence"].to_numpy()
        y = _to_logit(y_raw) if use_logit else y_raw
        color = provider_color(model_id)
        name = short_model(model_id)

        # Scatter points
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(size=4, color=color, opacity=0.3),
            showlegend=False,
            hoverinfo="skip",
        ))

        # OLS line
        reg = linregress(x, y)
        x_line = np.array([0.0, 1.0])
        y_line = reg.intercept + reg.slope * x_line
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(color=color, width=2.5),
            name=f"{name}  slope={reg.slope:.3f}",
        ))

    y_label = "Credence (logit)" if use_logit else "Credence"
    y_range = None if use_logit else [0, 1]
    fig.update_layout(
        height=350,
        xaxis=dict(title="User Valence", range=[-0.05, 1.05]),
        yaxis=dict(title=y_label, range=y_range),
        margin=dict(l=50, r=10, t=10, b=40),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_prompt_browser(prop_df: pl.DataFrame) -> None:
    """Render prompt selector and per-model responses."""
    # Group samples by prompt_text, compute avg credence + prompt attributes
    agg_exprs = [
        pl.col("consensus_credence").drop_nulls().mean().alias("avg_credence"),
        pl.col("consensus_credence").drop_nulls().count().alias("n_consensus"),
        pl.len().alias("n_models"),
    ]
    if "consensus_author_valence" in prop_df.columns:
        agg_exprs.append(pl.col("consensus_author_valence").first().alias("valence"))
    if "consensus_new_evidence_score" in prop_df.columns:
        agg_exprs.append(pl.col("consensus_new_evidence_score").first().alias("evidence"))

    prompt_groups = (
        prop_df.group_by("prompt_text")
        .agg(agg_exprs)
        .sort(["avg_credence", "prompt_text"], descending=[True, False])
    )

    if prompt_groups.is_empty():
        st.warning("No prompts found.")
        return

    prompts = prompt_groups["prompt_text"].to_list()
    avg_credences = prompt_groups["avg_credence"].to_list()

    prompt_to_label = {}
    for i, (p, avg) in enumerate(zip(prompts, avg_credences)):
        avg_str = f"{avg:.2f}" if avg is not None else "N/A"
        p_short = truncate(str(p).replace("\n", " "), 70)
        prompt_to_label[i] = f"{avg_str} — {p_short}"

    selected_idx = st.selectbox(
        "Prompt (by avg credence across models)",
        range(len(prompts)),
        format_func=lambda i: prompt_to_label[i],
        key="explore_inspect_prompt",
    )

    selected_prompt = prompts[selected_idx]

    # Show the full prompt
    st.markdown("**Prompt:**")
    st.code(str(selected_prompt)[:3000], language=None)

    # Prompt-level metrics
    row = prompt_groups.filter(pl.col("prompt_text") == selected_prompt).to_dicts()[0]
    metric_cols = st.columns(3)
    with metric_cols[0]:
        v = row.get("valence")
        st.metric("Prompt Valence", f"{v:.2f}" if v is not None else "N/A")
    with metric_cols[1]:
        e = row.get("evidence")
        st.metric("Information Score", f"{e:.2f}" if e is not None else "N/A")
    with metric_cols[2]:
        a = row.get("avg_credence")
        st.metric("Avg Credence", f"{a:.2f}" if a is not None else "N/A")

    # ── Step 3: Model responses ────────────────────────────────────────────
    prompt_samples = prop_df.filter(pl.col("prompt_text") == selected_prompt)
    models = sorted(prompt_samples["target_model"].unique().to_list())

    st.markdown(f"**Model Responses** ({len(models)} models)")

    for model_id in models:
        model_rows = prompt_samples.filter(pl.col("target_model") == model_id).to_dicts()
        model_name = short_model(model_id)
        color = provider_color(model_id)

        for s in model_rows:
            cc = s.get("consensus_credence")
            j1_c = s.get("judge1_credence")
            j2_c = s.get("judge2_credence")
            j1_inf = s.get("judge1_informative", True)
            j2_inf = s.get("judge2_informative", True)
            j1_ref = s.get("judge1_refusal", False)
            j2_ref = s.get("judge2_refusal", False)

            header = f"**{model_name}**"
            if cc is not None:
                header += f" — credence: {cc:.2f}"
            else:
                header += (
                    f" — No consensus "
                    f"(J1={_judge_str(j1_c, j1_inf, j1_ref)}, "
                    f"J2={_judge_str(j2_c, j2_inf, j2_ref)})"
                )

            with st.expander(header, expanded=True):
                st.markdown("**Response:**")
                st.code(str(s.get("response_text", ""))[:3000], language=None)

                j1_name = short_model(s["judge1_llm_id"]) if s.get("judge1_llm_id") else "Judge 1"
                j2_name = short_model(s["judge2_llm_id"]) if s.get("judge2_llm_id") else "Judge 2"

                col1, col2 = st.columns(2)
                with col1:
                    j1_label = f"{j1_c:.2f}" if j1_c is not None else "N/A"
                    if not j1_inf:
                        j1_label += " (refusal)" if j1_ref else " (non-informative)"
                    st.markdown(f"**{j1_name}**: {j1_label}")
                    st.text(str(s.get("judge1_explanation", ""))[:1000])
                with col2:
                    j2_label = f"{j2_c:.2f}" if j2_c is not None else "N/A"
                    if not j2_inf:
                        j2_label += " (refusal)" if j2_ref else " (non-informative)"
                    st.markdown(f"**{j2_name}**: {j2_label}")
                    st.text(str(s.get("judge2_explanation", ""))[:1000])


def _judge_str(credence, informative, refusal) -> str:
    """Format a single judge's result for the expander header."""
    if refusal:
        return "refusal"
    if not informative:
        return "non-inf"
    if credence is not None:
        return f"{credence:.2f}"
    return "N/A"
