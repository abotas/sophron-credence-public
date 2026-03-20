"""E2: Prompt Sensitivity — user valence correlation and credence distributions."""

from statistics import median

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from scipy.stats import pearsonr

from credence.viz.data import load_exploration
from credence.viz.formatting import model_sort_key, provider_color, short_model, unslugify
from credence.viz.stats import fisher_z_ci, format_shift, sig_stars


def render() -> None:
    """Render the Prompt Sensitivity tab."""
    df = load_exploration()
    if df.is_empty():
        st.info("No exploration data. Run `uvr python run_exploration.py` first.")
        return

    if "consensus_author_valence" not in df.columns:
        st.warning("Prompt attributes not yet scored. Run with prompt scoring enabled.")
        return

    # Domain filter
    dcol = "domain"
    st.markdown("**Domain filter**")
    all_domains = sorted(df[dcol].unique().to_list())
    selected_domains = st.multiselect(
        "Domain filter",
        all_domains,
        default=all_domains,
        format_func=unslugify,
        key="sensitivity_domains",
        label_visibility="collapsed",
    )

    if not selected_domains:
        st.info("Select at least one domain.")
        return

    filtered_df = df.filter(pl.col(dcol).is_in(selected_domains))
    selected_models = sorted(filtered_df["target_model"].unique().to_list(), key=model_sort_key)

    # Calculate % of neutral prompts
    total_prompts = len(filtered_df.filter(pl.col("consensus_author_valence").is_not_null()))
    neutral_prompts = len(filtered_df.filter(
        pl.col("consensus_author_valence").is_not_null()
        & ((pl.col("consensus_author_valence") - 0.5).abs() < 1e-9)
    ))
    neutral_pct = neutral_prompts / total_prompts * 100 if total_prompts > 0 else 0

    discard_neutral = st.session_state.get("sensitivity_discard_neutral", True)
    baseline_df = filtered_df
    if discard_neutral:
        baseline_df = baseline_df.filter((pl.col("consensus_author_valence") - 0.5).abs() > 1e-9)

    # ── Section 1: Correlation Forest Plot ────────────────────────────────
    st.markdown("**User Valence vs Credence Correlation**")
    st.caption(
        "Each prompt has a 'user valence' score (0-1) indicating whether the prompt "
        "suggests the user doubts (0) or believes (1) the proposition. This is correlated "
        "with the model's credence (also 0-1). A positive correlation means the model tends "
        "to express higher credence when the user seems to believe the proposition."
    )

    results = []
    for model_id in selected_models:
        model_data = baseline_df.filter(pl.col("target_model") == model_id)
        valid_data = (
            model_data.select(["consensus_credence", "consensus_author_valence"])
            .drop_nulls()
            .to_pandas()
        )
        if len(valid_data) < 10:
            continue

        x = valid_data["consensus_author_valence"].values
        y = valid_data["consensus_credence"].values
        r, p = pearsonr(x, y)
        ci = fisher_z_ci(r, len(x))

        results.append({
            "model_id": model_id,
            "model": short_model(model_id),
            "r": r,
            "p": p,
            "ci_low": ci[0],
            "ci_high": ci[1],
        })

    if results:
        import pandas as pd

        results_df = pd.DataFrame(results)

        fig_corr = go.Figure()
        for _, row in results_df.iterrows():
            color = provider_color(row["model_id"])
            sig = sig_stars(row["p"])
            fig_corr.add_trace(go.Scatter(
                x=[row["r"]],
                y=[row["model"]],
                mode="markers",
                marker=dict(size=10, color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[row["ci_high"] - row["r"]],
                    arrayminus=[row["r"] - row["ci_low"]],
                    thickness=2,
                    width=6,
                    color=color,
                ),
                hovertemplate=f"r = {row['r']:.3f}{sig}<br>95% CI: [{row['ci_low']:.3f}, {row['ci_high']:.3f}]<extra></extra>",
                showlegend=False,
            ))

        fig_corr.update_layout(
            height=max(200, len(results_df) * 50),
            margin=dict(l=10, r=10, t=10, b=55),
            xaxis=dict(title="Pearson r", range=[0, 1]),
            yaxis=dict(title=""),
            annotations=[
                dict(
                    text="Points = correlation coefficient. Whiskers = 95% CI.",
                    xref="paper", yref="paper",
                    x=0, y=-0.35,
                    xanchor="left", yanchor="top",
                    showarrow=False,
                    font=dict(size=10, color="#666"),
                )
            ],
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.checkbox(
            f"Discard neutral user valence (0.5) ({neutral_pct:.1f}% of prompts)",
            value=True,
            key="sensitivity_discard_neutral",
        )

    # ── Section 2: Distribution Histograms ────────────────────────────────
    st.markdown("**Credence Distributions**")
    st.caption(
        "Use the slider to filter prompts by user valence range. Compare how credence "
        "distributions shift when prompts imply the user doubts (low valence) vs believes "
        "(high valence) the proposition."
    )
    valence_range = st.slider(
        "User Valence",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05,
        key="sensitivity_valence",
    )

    attr_filtered_df = baseline_df
    if valence_range[0] > 0.0 or valence_range[1] < 1.0:
        attr_filtered_df = attr_filtered_df.filter(
            (pl.col("consensus_author_valence") >= valence_range[0] - 1e-9)
            & (pl.col("consensus_author_valence") <= valence_range[1] + 1e-9)
        )

    cols = st.columns(2)
    for i, model_id in enumerate(selected_models):
        baseline_credences = (
            baseline_df.filter(pl.col("target_model") == model_id)["consensus_credence"]
            .drop_nulls()
            .to_list()
        )
        credences = (
            attr_filtered_df.filter(pl.col("target_model") == model_id)["consensus_credence"]
            .drop_nulls()
            .to_list()
        )
        model_name = short_model(model_id)
        color = provider_color(model_id)

        n_baseline = len(baseline_credences)
        n_filtered = len(credences)
        baseline_mean = sum(baseline_credences) / n_baseline if baseline_credences else 0
        baseline_median = median(baseline_credences) if baseline_credences else 0

        fig = go.Figure()

        if credences:
            mean_cred = sum(credences) / len(credences)
            median_cred = median(credences)
            mean_shift = mean_cred - baseline_mean
            median_shift = median_cred - baseline_median

            fig.add_trace(go.Histogram(
                x=credences,
                xbins=dict(start=0, end=1, size=0.1),
                histnorm="percent",
                marker=dict(color=color, opacity=0.6),
                name="Filtered",
                showlegend=False,
            ))

            legend_text = (
                f"n={n_filtered}/{n_baseline}<br>"
                f"mean={mean_cred:.2f}{format_shift(mean_shift)}<br>"
                f"median={median_cred:.2f}{format_shift(median_shift)}"
            )
        else:
            legend_text = f"n=0/{n_baseline}"

        if baseline_credences:
            fig.add_trace(go.Histogram(
                x=baseline_credences,
                xbins=dict(start=0, end=1, size=0.1),
                histnorm="percent",
                marker=dict(color="rgba(0,0,0,0)", line=dict(color="#888888", width=1.5)),
                name="All",
                showlegend=False,
            ))

        fig.update_layout(
            height=160,
            title=model_name,
            title_font_size=12,
            xaxis=dict(range=[0, 1], title="Credence", title_font_size=10),
            yaxis=dict(title="%", title_font_size=10),
            margin=dict(l=40, r=90, t=35, b=30),
            bargap=0.1,
            barmode="overlay",
            annotations=[
                dict(
                    text=legend_text,
                    xref="paper", yref="paper",
                    x=1.02, y=0.95,
                    xanchor="left", yanchor="top",
                    showarrow=False,
                    font=dict(size=10),
                    align="left",
                )
            ],
        )

        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)

    # ── Section 3: User Valence Distribution ──────────────────────────────
    st.markdown("**User Valence Distribution**")
    valence_values = baseline_df["consensus_author_valence"].drop_nulls().to_list()
    if valence_values:
        bin_edges = [i * 0.05 for i in range(21)]
        counts, _ = np.histogram(valence_values, bins=bin_edges)
        pcts = [100 * c / len(valence_values) for c in counts]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(20)]

        fig_valence = go.Figure(go.Bar(
            x=bin_centers,
            y=pcts,
            width=0.045,
            marker_color="#636EFA",
        ))
        fig_valence.update_layout(
            height=120,
            margin=dict(l=40, r=10, t=10, b=30),
            xaxis=dict(range=[0, 1], title="User Valence", dtick=0.1),
            yaxis=dict(title="%"),
            bargap=0.1,
        )
        st.plotly_chart(fig_valence, use_container_width=True)
