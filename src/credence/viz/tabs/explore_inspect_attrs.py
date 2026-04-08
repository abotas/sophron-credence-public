"""Inspect Prompt Attributes tab — Deep-dive into prompt attribute judgments.

Flow: Select proposition → select prompt → view author valence and new evidence
judgments from each judge, alongside the credence consensus for context.
"""

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from credence.viz.data import load_exploration
from credence.viz.formatting import short_model, truncate, unslugify


def render() -> None:
    """Render the prompt attributes inspect tab."""
    st.subheader("Inspect: Prompt Attribute Judgments")
    st.caption(
        "Select a proposition, then pick a prompt to see how judges scored its "
        "author valence and new evidence content."
    )

    df = load_exploration()
    if df.is_empty():
        st.info("No exploration data. Run `uvr python run_exploration.py` first.")
        return

    has_valence = "prompt_judge1_author_valence" in df.columns
    has_evidence = "evidence_judge1_new_evidence_score" in df.columns

    if not has_valence and not has_evidence:
        st.warning("No prompt attribute data found. Run the prompt attributes pipeline first.")
        return

    # Attribute selector
    attr_options = []
    if has_valence:
        attr_options.append("Author Valence")
    if has_evidence:
        attr_options.append("New Evidence")

    selected_attr = st.radio(
        "Attribute", attr_options, horizontal=True, key="inspect_attrs_which",
    )
    show_valence = selected_attr == "Author Valence"
    show_evidence = selected_attr == "New Evidence"

    # Domain filter
    domains = sorted(df["domain"].unique().to_list())
    if len(domains) > 1:
        selected_domain = st.selectbox(
            "Domain", ["All"] + domains,
            format_func=lambda d: unslugify(d) if d != "All" else d,
            key="inspect_attrs_domain",
        )
        if selected_domain != "All":
            df = df.filter(pl.col("domain") == selected_domain)

    # Step 1: Select proposition
    prop_stats = (
        df.group_by("proposition")
        .agg([
            pl.col("consensus_credence").drop_nulls().median().alias("median_credence"),
            pl.len().alias("n_total"),
        ])
        .sort(["median_credence", "proposition"], descending=[True, False])
    )

    if prop_stats.is_empty():
        st.warning("No propositions found.")
        return

    props = prop_stats["proposition"].to_list()
    medians = prop_stats["median_credence"].to_list()

    prop_to_label = {}
    for p, m in zip(props, medians):
        m_str = f"{m:.2f}" if m is not None else "N/A"
        prop_to_label[p] = f"{m_str} — {truncate(p, 60)}"

    selected_prop = st.selectbox(
        "Proposition (by median credence)",
        props,
        format_func=lambda p: prop_to_label[p],
        key="inspect_attrs_prop",
    )

    prop_df = df.filter(pl.col("proposition") == selected_prop)
    st.markdown(f"**{selected_prop}**")

    # Overview charts
    _render_attribute_overview(prop_df, show_valence, show_evidence)

    st.divider()

    # Step 2: Prompt browser with attribute details
    _render_prompt_browser(prop_df, show_valence, show_evidence)


def _render_attribute_overview(
    prop_df: pl.DataFrame, show_valence: bool, show_evidence: bool
) -> None:
    """Render overview histogram for the selected prompt attribute."""
    if show_valence and "consensus_author_valence" in prop_df.columns:
        vals = prop_df["consensus_author_valence"].drop_nulls().to_list()
        if vals:
            fig = go.Figure(go.Histogram(
                x=vals,
                xbins=dict(start=0, end=1, size=0.05),
                marker_color="#636EFA", opacity=0.7,
            ))
            fig.update_layout(
                xaxis=dict(title="Author Valence", range=[0, 1]),
                yaxis=dict(title="Count"),
                height=250, margin=dict(t=10, b=30),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No consensus author valence values.")

    if show_evidence and "consensus_new_evidence_score" in prop_df.columns:
        vals = prop_df["consensus_new_evidence_score"].drop_nulls().to_list()
        if vals:
            fig = go.Figure(go.Histogram(
                x=vals,
                xbins=dict(start=0, end=1, size=0.05),
                marker_color="#EF553B", opacity=0.7,
            ))
            fig.update_layout(
                xaxis=dict(title="New Evidence Score", range=[0, 1]),
                yaxis=dict(title="Count"),
                height=250, margin=dict(t=10, b=30),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No consensus new evidence scores.")


def _render_prompt_browser(
    prop_df: pl.DataFrame, show_valence: bool, show_evidence: bool
) -> None:
    """Render prompt selector with attribute judgment details."""
    if "prompt_text" not in prop_df.columns:
        st.warning("Re-export with the latest exporter to populate prompt text.")
        return

    # Build per-prompt summary for the selector
    agg_exprs = [pl.len().alias("n_samples")]

    if show_valence and "consensus_author_valence" in prop_df.columns:
        agg_exprs.append(
            pl.col("consensus_author_valence").drop_nulls().mean().alias("avg_valence")
        )
    if show_evidence:
        # Use max across individual judge scores so we get a value even if only one judge scored
        evi_judge_cols = [c for c in prop_df.columns if c.startswith("evidence_judge") and c.endswith("_new_evidence_score")]
        if evi_judge_cols:
            max_expr = pl.max_horizontal(*[pl.col(c) for c in evi_judge_cols])
            agg_exprs.append(max_expr.mean().alias("avg_evidence"))
        elif "consensus_new_evidence_score" in prop_df.columns:
            agg_exprs.append(
                pl.col("consensus_new_evidence_score").drop_nulls().mean().alias("avg_evidence")
            )

    # Sort by the selected attribute's average value
    sort_col = "avg_valence" if show_valence else "avg_evidence"

    prompt_groups = (
        prop_df.group_by("prompt_text")
        .agg(agg_exprs)
        .sort([sort_col, "prompt_text"], descending=[True, False], nulls_last=True)
    )

    if prompt_groups.is_empty():
        st.warning("No prompts found.")
        return

    prompts = prompt_groups["prompt_text"].to_list()

    # Build labels showing only the selected attribute
    prompt_to_label = {}
    for i, row in enumerate(prompt_groups.iter_rows(named=True)):
        if show_valence and "avg_valence" in row and row["avg_valence"] is not None:
            prefix = f"{row['avg_valence']:.2f}"
        elif show_evidence and "avg_evidence" in row and row["avg_evidence"] is not None:
            prefix = f"{row['avg_evidence']:.2f}"
        else:
            prefix = "N/A"
        p_short = truncate(str(row["prompt_text"]).replace("\n", " "), 60)
        prompt_to_label[i] = f"{prefix} — {p_short}"

    selected_idx = st.selectbox(
        "Prompt",
        range(len(prompts)),
        format_func=lambda i: prompt_to_label[i],
        key="inspect_attrs_prompt",
    )

    selected_prompt = prompts[selected_idx]

    # Show full prompt text
    st.markdown("**Prompt:**")
    st.code(str(selected_prompt)[:3000], language=None)

    # Prompt attributes are per-prompt (not per-model), so just take the first row
    prompt_samples = prop_df.filter(pl.col("prompt_text") == selected_prompt)
    if prompt_samples.is_empty():
        return

    row = prompt_samples.row(0, named=True)

    if show_valence:
        _render_valence_judges(row)

    if show_evidence:
        _render_evidence_judges(row)


def _render_valence_judges(row: dict) -> None:
    """Render author valence judge results side by side."""
    # Discover judge columns
    judge_indices = []
    i = 1
    while f"prompt_judge{i}_author_valence" in row:
        judge_indices.append(i)
        i += 1

    if not judge_indices:
        st.caption("No author valence judgments.")
        return

    consensus = row.get("consensus_author_valence")
    if consensus is not None:
        st.markdown(f"Consensus: **{consensus:.2f}**")

    cols = st.columns(len(judge_indices))
    for col, idx in zip(cols, judge_indices):
        with col:
            judge_id = row.get(f"prompt_judge{idx}_llm_id", f"Judge {idx}")
            valence = row.get(f"prompt_judge{idx}_author_valence")
            explanation = row.get(f"prompt_judge{idx}_explanation", "")

            val_str = f"{valence:.2f}" if valence is not None else "N/A"
            st.markdown(f"**{short_model(judge_id)}**: {val_str}")
            if explanation:
                st.text(str(explanation)[:1000])


def _render_evidence_judges(row: dict) -> None:
    """Render new evidence judge results side by side."""
    judge_indices = []
    i = 1
    while f"evidence_judge{i}_new_evidence_score" in row:
        judge_indices.append(i)
        i += 1

    if not judge_indices:
        st.caption("No new evidence judgments.")
        return

    scores = [row.get(f"evidence_judge{idx}_new_evidence_score") for idx in judge_indices]
    scores = [s for s in scores if s is not None]
    if scores:
        st.markdown(f"Max score: **{max(scores):.2f}**")

    cols = st.columns(len(judge_indices))
    for col, idx in zip(cols, judge_indices):
        with col:
            judge_id = row.get(f"evidence_judge{idx}_llm_id", f"Judge {idx}")
            score = row.get(f"evidence_judge{idx}_new_evidence_score")
            direction = row.get(f"evidence_judge{idx}_direction", "")
            evidence_id = row.get(f"evidence_judge{idx}_evidence_identified", "")
            explanation = row.get(f"evidence_judge{idx}_explanation", "")

            score_str = f"{score:.2f}" if score is not None else "N/A"
            st.markdown(f"**{short_model(judge_id)}**: {score_str} ({direction})")

            if evidence_id:
                st.markdown(f"*Evidence identified:* {str(evidence_id)[:500]}")
            if explanation:
                st.text(str(explanation)[:1000])
