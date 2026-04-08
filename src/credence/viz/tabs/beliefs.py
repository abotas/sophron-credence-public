"""E1: What Models Believe — extremity, dispersion, and deep dive."""

from statistics import median

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from credence.viz.constants import DOMAIN_COLORS
from credence.viz.data import load_exploration
from credence.viz.formatting import provider_color, short_model, unslugify


def render() -> None:
    """Render the What Models Believe tab."""
    df = load_exploration()
    if df.is_empty():
        st.info("No exploration data. Run `uvr python run_exploration.py` first.")
        return

    col_title, col_toggle = st.columns([3, 2])
    with col_title:
        st.subheader("What Models Believe")
    with col_toggle:
        view_mode = st.segmented_control(
            "View",
            ["Overview", "Deep Dive"],
            default="Overview",
            key="beliefs_view_mode",
            label_visibility="collapsed",
        )

    if view_mode == "Overview":
        _render_overview(df)
    else:
        _render_deep_dive(df)


def _render_overview(df: pl.DataFrame) -> None:
    """Render extremity and dispersion overview charts."""
    # ── Extremity ─────────────────────────────────────────────────────────
    st.markdown(
        "**Extremity** — How far from 0.5 are model credences? "
        "Higher values indicate stronger positions; lower values indicate more hedging."
    )

    dcol = "domain"

    extremity_data = (
        df.filter(pl.col("consensus_credence").is_not_null())
        .group_by(["target_model", "proposition", dcol])
        .agg(pl.col("consensus_credence").mean().alias("mean_credence"))
    )

    if extremity_data.is_empty():
        st.warning("No data for extremity calculation.")
        return

    ext_pdf = extremity_data.to_pandas()
    ext_pdf["extremity"] = (ext_pdf["mean_credence"] - 0.5).abs()
    ext_pdf["model"] = ext_pdf["target_model"].apply(short_model)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### By Model")
        model_stats = (
            ext_pdf.groupby(["target_model", "model"])["extremity"]
            .mean()
            .reset_index()
            .rename(columns={"extremity": "mean_extremity"})
            .sort_values("mean_extremity", ascending=True)
        )
        model_stats["color"] = model_stats["target_model"].apply(provider_color)

        fig = go.Figure(go.Bar(
            y=model_stats["model"],
            x=model_stats["mean_extremity"],
            orientation="h",
            marker_color=model_stats["color"],
            text=[f"{v:.2f}" for v in model_stats["mean_extremity"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=max(250, len(model_stats) * 35),
            margin=dict(l=10, r=40, t=10, b=40),
            xaxis=dict(title="Mean Extremity", range=[0, model_stats["mean_extremity"].max() * 1.25]),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("### By Domain")
        dom_stats = (
            ext_pdf.groupby(dcol)["extremity"]
            .mean()
            .reset_index()
            .rename(columns={"extremity": "mean_extremity"})
            .sort_values("mean_extremity", ascending=True)
        )
        dom_stats["display"] = dom_stats[dcol].apply(lambda x: x.replace("_", " ").title())
        dom_stats["color"] = dom_stats[dcol].apply(lambda x: DOMAIN_COLORS.get(x, "#636EFA"))

        fig = go.Figure(go.Bar(
            y=dom_stats["display"],
            x=dom_stats["mean_extremity"],
            orientation="h",
            marker_color=dom_stats["color"],
            text=[f"{v:.2f}" for v in dom_stats["mean_extremity"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=max(250, len(dom_stats) * 35),
            margin=dict(l=10, r=40, t=10, b=40),
            xaxis=dict(title="Mean Extremity", range=[0, dom_stats["mean_extremity"].max() * 1.25]),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig, width="stretch")

    # ── Dispersion ────────────────────────────────────────────────────────
    st.markdown(
        "**Dispersion** — How much does credence vary across prompts for the same "
        "proposition? Higher IQR indicates more sensitivity to prompt framing."
    )

    iqr_data = (
        df.filter(pl.col("consensus_credence").is_not_null())
        .group_by(["target_model", "proposition", dcol])
        .agg([
            (pl.col("consensus_credence").quantile(0.75) - pl.col("consensus_credence").quantile(0.25)).alias("iqr"),
            pl.col("consensus_credence").len().alias("n_samples"),
        ])
        .filter(pl.col("n_samples") >= 5)
    )

    if iqr_data.is_empty():
        st.warning("Not enough data per proposition to compute IQR.")
        return

    iqr_pdf = iqr_data.to_pandas()
    iqr_pdf["model"] = iqr_pdf["target_model"].apply(short_model)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### By Model")
        model_stats = (
            iqr_pdf.groupby(["target_model", "model"])["iqr"]
            .mean()
            .reset_index()
            .rename(columns={"iqr": "mean_iqr"})
            .sort_values("mean_iqr", ascending=True)
        )
        model_stats["color"] = model_stats["target_model"].apply(provider_color)

        fig = go.Figure(go.Bar(
            y=model_stats["model"],
            x=model_stats["mean_iqr"],
            orientation="h",
            marker_color=model_stats["color"],
            text=[f"{v:.2f}" for v in model_stats["mean_iqr"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=max(250, len(model_stats) * 35),
            margin=dict(l=10, r=40, t=10, b=40),
            xaxis=dict(title="Mean IQR", range=[0, model_stats["mean_iqr"].max() * 1.25]),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("### By Domain")
        dom_stats = (
            iqr_pdf.groupby(dcol)["iqr"]
            .mean()
            .reset_index()
            .rename(columns={"iqr": "mean_iqr"})
            .sort_values("mean_iqr", ascending=True)
        )
        dom_stats["display"] = dom_stats[dcol].apply(lambda x: x.replace("_", " ").title())
        dom_stats["color"] = dom_stats[dcol].apply(lambda x: DOMAIN_COLORS.get(x, "#636EFA"))

        fig = go.Figure(go.Bar(
            y=dom_stats["display"],
            x=dom_stats["mean_iqr"],
            orientation="h",
            marker_color=dom_stats["color"],
            text=[f"{v:.2f}" for v in dom_stats["mean_iqr"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=max(250, len(dom_stats) * 35),
            margin=dict(l=10, r=40, t=10, b=40),
            xaxis=dict(title="Mean IQR", range=[0, dom_stats["mean_iqr"].max() * 1.25]),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig, width="stretch")


def _render_deep_dive(df: pl.DataFrame) -> None:
    """Render proposition-level deep dive."""
    # Domain selector
    dcol = "domain"
    all_domains = sorted(df[dcol].unique().to_list())
    selected_domain = st.selectbox("Domain", all_domains, format_func=unslugify, key="beliefs_domain")
    cat_df = df.filter(pl.col(dcol) == selected_domain)

    # Model selector
    all_models = sorted(cat_df["target_model"].unique().to_list())
    all_model_names = [short_model(m) for m in all_models]
    name_to_id = {short_model(m): m for m in all_models}

    selected_model_names = st.multiselect(
        "Models", all_model_names, default=all_model_names, key="beliefs_models"
    )
    selected_models = [name_to_id[n] for n in selected_model_names]

    if selected_models:
        cat_df = cat_df.filter(pl.col("target_model").is_in(selected_models))

    # Proposition stats sorted by median credence
    prop_stats = (
        cat_df.group_by("proposition")
        .agg([
            pl.col("consensus_credence").drop_nulls().median().alias("median"),
            pl.len().alias("n_total"),
        ])
        .sort(["median", "proposition"], descending=[True, False])
    )

    if prop_stats.is_empty():
        st.warning("No propositions match the selected filters.")
        return

    props = prop_stats["proposition"].to_list()
    medians = prop_stats["median"].to_list()

    def format_prop(p: str) -> str:
        idx = props.index(p)
        m = medians[idx]
        prefix = f"{m:.2f}" if m is not None else "N/A"
        truncated = p[:55] + "..." if len(p) > 55 else p
        return f"{prefix} - {truncated}"

    selected_prop = st.selectbox("Proposition", props, format_func=format_prop, key="beliefs_prop")
    st.caption(selected_prop)

    # Histogram colored by model
    prop_df = cat_df.filter(pl.col("proposition") == selected_prop)

    fig = go.Figure()
    for model_id in selected_models:
        credences = (
            prop_df.filter(pl.col("target_model") == model_id)["consensus_credence"]
            .drop_nulls()
            .to_list()
        )
        if credences:
            model_median = median(credences)
            fig.add_trace(go.Histogram(
                x=credences,
                xbins=dict(start=0, end=1, size=0.05),
                name=f"{short_model(model_id)} (med: {model_median:.2f})",
                marker_color=provider_color(model_id),
                opacity=0.7,
            ))

    fig.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=10, b=30),
        xaxis=dict(range=[0, 1], title="Credence"),
        yaxis=dict(title=None),
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # Consensus / non-informative / refusal breakdown
    def _categorize(row: dict) -> str:
        if row.get("consensus_credence") is not None:
            return "consensus"
        if row.get("judge1_refusal") or row.get("judge2_refusal"):
            return "refusal"
        return "non_informative"

    samples = prop_df.to_dicts()
    model_counts = {m: {"consensus": 0, "non_informative": 0, "refusal": 0} for m in selected_models}
    for s in samples:
        mid = s.get("target_model", "")
        if mid in model_counts:
            model_counts[mid][_categorize(s)] += 1

    models_display = [short_model(m) for m in selected_models]
    model_totals = {m: sum(model_counts[m].values()) for m in selected_models}

    def pct(count: int, total: int) -> float:
        return 100 * count / total if total > 0 else 0

    fig_counts = go.Figure()
    fig_counts.add_trace(go.Bar(
        name="Consensus",
        x=models_display,
        y=[pct(model_counts[m]["consensus"], model_totals[m]) for m in selected_models],
        marker_color="#2ca02c",
        text=[f"{pct(model_counts[m]['consensus'], model_totals[m]):.0f}%" for m in selected_models],
        textposition="auto",
    ))
    fig_counts.add_trace(go.Bar(
        name="Non-informative",
        x=models_display,
        y=[pct(model_counts[m]["non_informative"], model_totals[m]) for m in selected_models],
        marker_color="#ff7f0e",
        text=[f"{pct(model_counts[m]['non_informative'], model_totals[m]):.0f}%" for m in selected_models],
        textposition="auto",
    ))
    fig_counts.add_trace(go.Bar(
        name="Refusal",
        x=models_display,
        y=[pct(model_counts[m]["refusal"], model_totals[m]) for m in selected_models],
        marker_color="#d62728",
        text=[f"{pct(model_counts[m]['refusal'], model_totals[m]):.0f}%" for m in selected_models],
        textposition="auto",
    ))
    fig_counts.update_layout(
        height=150,
        margin=dict(l=0, r=0, t=10, b=30),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="%", range=[0, 105]),
    )
    st.plotly_chart(fig_counts, width="stretch")

    st.divider()

    # Expandable sample cards
    def _sort_key(s: dict) -> tuple:
        cc = s.get("consensus_credence")
        return (0, -cc) if cc is not None else (1, 0)

    sorted_samples = sorted(samples, key=_sort_key)
    st.markdown(f"**Samples** ({len(sorted_samples)} total)")

    for s in sorted_samples:
        cc = s.get("consensus_credence")
        model_name = s.get("target_model", "unknown").split("/")[-1]
        prompt_preview = s.get("prompt_text", "")[:80].replace("\n", " ")
        if len(s.get("prompt_text", "")) > 80:
            prompt_preview += "..."

        j1_c = s.get("judge1_credence")
        j2_c = s.get("judge2_credence")
        j1_inf = s.get("judge1_informative", True)
        j2_inf = s.get("judge2_informative", True)
        j1_ref = s.get("judge1_refusal", False)
        j2_ref = s.get("judge2_refusal", False)

        def judge_str(credence, informative, refusal):
            if informative and credence is not None:
                return f"{credence:.2f}"
            return "Refusal" if refusal else "non-inf"

        header = f"**{model_name}** | "
        if cc is not None:
            header += f"{cc:.2f}"
        else:
            header += f"No consensus (J1={judge_str(j1_c, j1_inf, j1_ref)}, J2={judge_str(j2_c, j2_inf, j2_ref)})"
        header += f" | {prompt_preview}"

        with st.expander(header, expanded=False):
            st.markdown("**Prompt:**")
            st.code(s.get("prompt_text", "")[:1000], language=None)
            st.markdown("**Response:**")
            st.code(s.get("response_text", "")[:1500], language=None)

            j1_name = s.get("judge1_llm_id", "Judge 1").split("/")[-1]
            j2_name = s.get("judge2_llm_id", "Judge 2").split("/")[-1]

            j1_label = f"{j1_c:.2f}" if j1_c is not None else "N/A"
            if not j1_inf:
                j1_label += " (Refusal)" if j1_ref else " (non-informative)"
            st.markdown(f"**{j1_name}**: {j1_label}")
            st.text(s.get("judge1_explanation", "")[:500])

            j2_label = f"{j2_c:.2f}" if j2_c is not None else "N/A"
            if not j2_inf:
                j2_label += " (Refusal)" if j2_ref else " (non-informative)"
            st.markdown(f"**{j2_name}**: {j2_label}")
            st.text(s.get("judge2_explanation", "")[:500])
