"""E3: Cross-Model Agreement — pairwise heatmap, domain strip plot, disagreement outliers."""

import math

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from scipy.stats import pearsonr, spearmanr

from credence.viz.data import load_exploration
from credence.viz.formatting import provider_color, short_model, unslugify


def render() -> None:
    """Render the Cross-Model Agreement tab."""
    st.subheader("Model Agreement")

    df = load_exploration()
    if df.is_empty():
        st.info("No exploration data. Run `uvr python run_exploration.py` first.")
        return

    dcol = "domain"
    all_models = sorted(df["target_model"].unique().to_list())
    all_model_names = [short_model(m) for m in all_models]
    name_to_id = {short_model(m): m for m in all_models}
    all_domains = sorted(df[dcol].unique().to_list())

    # Filters
    with st.expander("Filters", expanded=False):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            selected_model_names = st.multiselect(
                "Models", all_model_names, default=all_model_names, key="agreement_models"
            )
        with col2:
            selected_domains = st.multiselect(
                "Domains", all_domains, default=all_domains,
                format_func=unslugify, key="agreement_domains",
            )
        with col3:
            metric_type = st.radio(
                "Metric", ["Pearson", "Spearman", "MAE"], horizontal=True, key="agreement_metric_type"
            )

    use_mae = metric_type == "MAE"
    selected_models = [name_to_id[n] for n in selected_model_names if n in name_to_id]

    filtered_df = df
    if selected_models:
        filtered_df = filtered_df.filter(pl.col("target_model").is_in(selected_models))
    if selected_domains:
        filtered_df = filtered_df.filter(pl.col(dcol).is_in(selected_domains))

    if len(selected_models) < 2:
        st.warning("Select at least 2 models to compare.")
        return

    # Pivot: credences per sample_id per model
    pivot_pdf = (
        filtered_df.filter(pl.col("consensus_credence").is_not_null())
        .select(["sample_id", "target_model", "consensus_credence"])
        .to_pandas()
    )

    if len(pivot_pdf) == 0:
        st.warning("No consensus credences found.")
        return

    pivot_wide = pivot_pdf.pivot(index="sample_id", columns="target_model", values="consensus_credence")

    # Pairwise metric matrix
    n = len(selected_models)
    matrix = np.full((n, n), np.nan)

    for i, m1 in enumerate(selected_models):
        for j, m2 in enumerate(selected_models):
            if i == j:
                matrix[i, j] = 0.0 if use_mae else 1.0
            elif m1 in pivot_wide.columns and m2 in pivot_wide.columns:
                valid = pivot_wide[[m1, m2]].dropna()
                if len(valid) >= 10:
                    if use_mae:
                        matrix[i, j] = np.abs(valid[m1].values - valid[m2].values).mean()
                    else:
                        corr_fn = spearmanr if metric_type == "Spearman" else pearsonr
                        matrix[i, j] = corr_fn(valid[m1].values, valid[m2].values)[0]

    friendly = [short_model(m) for m in selected_models]

    # Collect unique pairs for summary
    pair_metrics = []
    for i in range(n):
        for j in range(i + 1, n):
            if not math.isnan(matrix[i, j]):
                pair_metrics.append({
                    "Model 1": friendly[i],
                    "Model 2": friendly[j],
                    "value": matrix[i, j],
                })
    pair_metrics.sort(key=lambda x: x["value"], reverse=not use_mae)

    if pair_metrics:
        values = [p["value"] for p in pair_metrics]
        z_min, z_max = min(values) - 0.02, max(values) + 0.02
    else:
        z_min, z_max = (0, 0.5) if use_mae else (0, 1)

    text_labels = [
        [f"{matrix[i, j]:.2f}" if not math.isnan(matrix[i, j]) else "" for j in range(n)]
        for i in range(n)
    ]

    colorscale = "RdBu" if use_mae else "RdBu_r"
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=matrix,
        x=friendly,
        y=friendly,
        text=text_labels,
        texttemplate="%{text}",
        colorscale=colorscale,
        zmin=z_min,
        zmax=z_max,
        colorbar=dict(title="MAE" if use_mae else "r"),
    ))
    fig_heatmap.update_layout(
        height=400,
        margin=dict(l=120, r=20, t=20, b=120),
        xaxis=dict(tickangle=45),
    )

    # ── Heatmap ───────────────────────────────────────────────────────────
    st.markdown("**Pairwise Model Agreement**")
    if use_mae:
        st.caption("Mean absolute error between model credences across all propositions. Lower = more similar.")
    else:
        st.caption(f"{metric_type} correlation between model credences. Higher = more agreement.")
    st.plotly_chart(fig_heatmap, width="stretch")

    if pair_metrics:
        best, worst = pair_metrics[0], pair_metrics[-1]
        label = "MAE" if use_mae else "r"
        st.caption(
            f"Most similar: **{best['Model 1']}** & **{best['Model 2']}** ({label}={best['value']:.2f}) | "
            f"Least similar: **{worst['Model 1']}** & **{worst['Model 2']}** ({label}={worst['value']:.2f})"
        )

    # ── Domain strip plot ─────────────────────────────────────────────────
    if all_domains and selected_domains:
        _render_domain_strip(filtered_df, selected_models, selected_domains, dcol, metric_type, use_mae)

    # ── Disagreement outliers ─────────────────────────────────────────────
    _render_disagreement_outliers(filtered_df, selected_models, dcol)


def _render_domain_strip(
    filtered_df: pl.DataFrame,
    selected_models: list[str],
    selected_domains: list[str],
    dcol: str,
    metric_type: str,
    use_mae: bool,
) -> None:
    """Render the domain strip plot."""
    domain_points = []
    domain_means: dict[str, float] = {}

    for cat in selected_domains:
        cat_df = filtered_df.filter(pl.col(dcol) == cat)
        cat_pivot = (
            cat_df.filter(pl.col("consensus_credence").is_not_null())
            .select(["sample_id", "target_model", "consensus_credence"])
            .to_pandas()
        )
        if len(cat_pivot) < 10:
            continue

        cat_wide = cat_pivot.pivot(index="sample_id", columns="target_model", values="consensus_credence")
        cat_values = []

        for i, m1 in enumerate(selected_models):
            for j, m2 in enumerate(selected_models):
                if i < j and m1 in cat_wide.columns and m2 in cat_wide.columns:
                    valid = cat_wide[[m1, m2]].dropna()
                    if len(valid) >= 10:
                        if use_mae:
                            val = np.abs(valid[m1].values - valid[m2].values).mean()
                        else:
                            corr_fn = spearmanr if metric_type == "Spearman" else pearsonr
                            val = corr_fn(valid[m1].values, valid[m2].values)[0]
                        cat_values.append(val)
                        domain_points.append({
                            "domain": cat,
                            "value": val,
                            "model1": short_model(m1),
                            "model2": short_model(m2),
                        })

        if cat_values:
            domain_means[cat] = sum(cat_values) / len(cat_values)

    if not domain_points:
        return

    import pandas as pd

    points_df = pd.DataFrame(domain_points)
    sorted_cats = sorted(domain_means.keys(), key=lambda d: domain_means[d], reverse=not use_mae)

    fig_strip = go.Figure()
    metric_label = "MAE" if use_mae else "r"

    for cat in sorted_cats:
        cat_data = points_df[points_df["domain"] == cat]
        vals = cat_data["value"].values
        m1s = cat_data["model1"].values
        m2s = cat_data["model2"].values
        jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(vals))
        y_pos = [sorted_cats.index(cat) + j for j in jitter]

        fig_strip.add_trace(go.Scattergl(
            x=vals,
            y=y_pos,
            mode="markers",
            marker=dict(size=8, opacity=0.6),
            name=unslugify(cat),
            showlegend=False,
            customdata=list(zip(m1s, m2s)),
            hovertemplate=f"%{{customdata[0]}} & %{{customdata[1]}}<br>{metric_label}=%{{x:.2f}}<extra></extra>",
        ))
        fig_strip.add_trace(go.Scatter(
            x=[domain_means[cat]],
            y=[sorted_cats.index(cat)],
            mode="markers",
            marker=dict(size=14, symbol="diamond", color="black", line=dict(width=2, color="white")),
            showlegend=False,
            hovertemplate="Mean: %{x:.2f}<extra></extra>",
        ))

    x_title = "Mean Absolute Error" if use_mae else f"Pairwise Correlation ({metric_label})"
    x_range = [0, max(points_df["value"]) + 0.05] if use_mae else [0, 1]
    fig_strip.update_layout(
        height=400,
        margin=dict(l=120, r=20, t=20, b=40),
        xaxis=dict(title=x_title, range=x_range),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(sorted_cats))),
            ticktext=[unslugify(c) for c in sorted_cats],
            title="",
        ),
    )

    st.markdown("**Agreement by Category**")
    st.caption("Same metric computed per category. Each dot = one model pair. Diamonds = category mean.")
    st.plotly_chart(fig_strip, width="stretch")


def _render_disagreement_outliers(
    filtered_df: pl.DataFrame,
    selected_models: list[str],
    dcol: str,
) -> None:
    """Render disagreement outlier propositions."""
    st.markdown("**Disagreement Outliers**")
    st.caption(
        "Propositions ranked by cross-model disagreement. For each proposition, "
        "we compute each model's median credence, then measure the standard deviation "
        "across models. High std = models disagree."
    )

    prop_model_medians = (
        filtered_df.filter(pl.col("consensus_credence").is_not_null())
        .group_by(["proposition", dcol, "target_model"])
        .agg(pl.col("consensus_credence").median().alias("model_median"))
    )

    prop_disagreement = (
        prop_model_medians.group_by(["proposition", dcol])
        .agg([
            pl.col("model_median").std().alias("std"),
            pl.col("model_median").min().alias("min"),
            pl.col("model_median").max().alias("max"),
            pl.col("model_median").median().alias("overall_median"),
            pl.len().alias("n_models"),
        ])
        .filter(pl.col("n_models") >= 2)
        .sort("std", descending=True)
    )

    all_disagreement = prop_disagreement.to_dicts()
    total_props = len(all_disagreement)
    page_size = 5
    total_pages = max(1, (total_props + page_size - 1) // page_size)

    if not all_disagreement:
        st.info("Need at least 2 models evaluated on the same propositions.")
        return

    if "disagreement_page" not in st.session_state or not isinstance(st.session_state.disagreement_page, int):
        st.session_state.disagreement_page = 0

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Prev", disabled=st.session_state.disagreement_page == 0, key="disagreement_prev"):
            st.session_state.disagreement_page -= 1
            st.rerun()
    with col2:
        start = st.session_state.disagreement_page * page_size + 1
        end = min(start + page_size - 1, total_props)
        st.markdown(f"**{start}-{end}** of {total_props}")
    with col3:
        if st.button("Next", disabled=st.session_state.disagreement_page >= total_pages - 1, key="disagreement_next"):
            st.session_state.disagreement_page += 1
            st.rerun()

    start_idx = st.session_state.disagreement_page * page_size
    page_props = all_disagreement[start_idx:start_idx + page_size]

    for rank, prop_stats in enumerate(page_props, start=start_idx + 1):
        prop = prop_stats["proposition"]
        std = prop_stats["std"]

        st.markdown(f"**{rank}. [std={std:.2f}]** {prop}")

        prop_data = (
            prop_model_medians.filter(pl.col("proposition") == prop)
            .sort("model_median", descending=True)
            .to_dicts()
        )

        fig_dots = go.Figure()
        fig_dots.add_shape(
            type="line",
            x0=0, x1=1, y0=0, y1=0,
            line=dict(color="gray", width=2),
        )

        sorted_data = sorted(prop_data, key=lambda d: d["model_median"])

        # Stagger labels to avoid overlap
        y_levels = [20, 45, 70, 95]
        label_positions = []
        for d in sorted_data:
            level = 0
            for prev_d, prev_level in label_positions:
                if abs(d["model_median"] - prev_d["model_median"]) < 0.12:
                    if prev_level >= level:
                        level = prev_level + 1
            label_positions.append((d, level % len(y_levels)))

        for d, level in label_positions:
            model_id = d["target_model"]
            color = provider_color(model_id)
            fig_dots.add_trace(go.Scatter(
                x=[d["model_median"]],
                y=[0],
                mode="markers",
                marker=dict(size=12, color=color),
                hovertemplate=f"{short_model(model_id)}: %{{x:.2f}}<extra></extra>",
                showlegend=False,
            ))
            fig_dots.add_annotation(
                x=d["model_median"],
                y=0,
                text=short_model(model_id),
                textangle=-45,
                showarrow=True,
                arrowhead=0,
                arrowwidth=1,
                arrowcolor="#ccc",
                ay=-y_levels[level],
                ax=0,
                font=dict(size=10, color=color),
            )

        fig_dots.update_layout(
            height=180,
            margin=dict(l=20, r=20, t=80, b=20),
            xaxis=dict(title="", range=[-0.02, 1.02], dtick=0.2),
            yaxis=dict(visible=False, range=[-0.3, 0.3]),
            showlegend=False,
        )
        st.plotly_chart(fig_dots, width="stretch", key=f"disagreement_plot_{rank}")
