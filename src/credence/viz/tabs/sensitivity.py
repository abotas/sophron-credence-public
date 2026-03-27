"""E2: Prompt Sensitivity — user valence correlation and credence distributions."""

from statistics import median

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from scipy.stats import linregress, pearsonr, t as t_dist

from credence.viz.data import load_exploration
from credence.viz.formatting import model_sort_key, provider_color, short_model, unslugify
from credence.core.util import provider
from credence.viz.stats import fisher_z_ci, format_shift, sig_stars


def _demean_within_proposition(
    x: np.ndarray, y: np.ndarray, props: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Demean x and y within each proposition, dropping singletons."""
    x_dm = np.empty_like(x)
    y_dm = np.empty_like(y)
    keep = np.zeros(len(x), dtype=bool)
    for prop in np.unique(props):
        mask = props == prop
        if mask.sum() >= 2:
            x_dm[mask] = x[mask] - x[mask].mean()
            y_dm[mask] = y[mask] - y[mask].mean()
            keep |= mask
    return x_dm[keep], y_dm[keep]


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

    # New evidence filter — exclude prompts where any judge scored evidence > 0.4
    evidence_cols = [c for c in filtered_df.columns if c.endswith("_new_evidence_score") and c.startswith("evidence_judge")]
    if evidence_cols:
        check_col, count_col = st.columns([3, 1])
        with check_col:
            exclude_evidence = st.checkbox(
                "Exclude prompts with new evidence (any judge > 0.4)",
                value=True,
                key="sensitivity_exclude_evidence",
            )
        if exclude_evidence:
            any_null = pl.lit(False)
            any_above = pl.lit(False)
            for c in evidence_cols:
                any_null = any_null | pl.col(c).is_null()
                any_above = any_above | (pl.col(c) > 0.4 + 1e-9)
            pre_filter_n = len(filtered_df)
            filtered_df = filtered_df.filter(~any_null & ~any_above)
            excluded_n = pre_filter_n - len(filtered_df)
            with count_col:
                st.caption(f"Excluded {excluded_n} prompts")
    selected_models = sorted(filtered_df["target_model"].unique().to_list(), key=model_sort_key)

    # Calculate % of neutral prompts
    total_prompts = len(filtered_df.filter(pl.col("consensus_author_valence").is_not_null()))
    neutral_prompts = len(filtered_df.filter(
        pl.col("consensus_author_valence").is_not_null()
        & ((pl.col("consensus_author_valence") - 0.5).abs() < 1e-9)
    ))
    neutral_pct = neutral_prompts / total_prompts * 100 if total_prompts > 0 else 0

    discard_neutral = st.checkbox(
        f"Discard neutral user valence (0.5) ({neutral_pct:.1f}% of prompts)",
        value=True,
        key="sensitivity_discard_neutral",
    )
    baseline_df = filtered_df
    if discard_neutral:
        baseline_df = baseline_df.filter((pl.col("consensus_author_valence") - 0.5).abs() > 1e-9)

    use_logit = st.checkbox("Logit scale", value=True, key="sensitivity_logit_scale")
    fixed_effects = st.checkbox(
        "Within-proposition (fixed effects)",
        value=True,
        key="sensitivity_fixed_effects",
        help="Demean valence and credence within each proposition to isolate sycophancy from between-proposition base-rate differences.",
    )

    # ── Section 1: Sycophancy Regression ───────────────────────────────
    st.markdown("**Sycophancy Regression — User Valence → Credence**")
    st.caption(
        "Scatter plots of user valence (x) vs model credence (y) with OLS regression lines. "
        "The slope measures how much the model's expressed credence shifts across the full "
        "range of user valence — a steeper slope indicates more sycophantic behavior."
    )

    def _to_logit(arr):
        clipped = np.clip(arr, 0.01, 0.99)
        return np.log(clipped / (1 - clipped))

    # Assign dash styles: models sharing a provider get the same color,
    # distinguished by dash pattern (solid, dash, dot, dashdot).
    _DASH_STYLES = ["solid", "dash", "dot", "dashdot"]
    _provider_counter: dict[str, int] = {}
    model_dash: dict[str, str] = {}
    for model_id in selected_models:
        prov = provider(model_id)
        idx = _provider_counter.get(prov, 0)
        model_dash[model_id] = _DASH_STYLES[idx % len(_DASH_STYLES)]
        _provider_counter[prov] = idx + 1

    slope_results = []
    fig_scatter = go.Figure()

    for model_id in selected_models:
        model_data = baseline_df.filter(pl.col("target_model") == model_id)
        sel_cols = ["consensus_credence", "consensus_author_valence"]
        if fixed_effects:
            sel_cols.append("proposition")
        valid = model_data.select(sel_cols).drop_nulls().to_pandas()
        if len(valid) < 10:
            continue

        x = valid["consensus_author_valence"].values
        y_raw = valid["consensus_credence"].values
        y = _to_logit(y_raw) if use_logit else y_raw

        if fixed_effects:
            x, y = _demean_within_proposition(x, y, valid["proposition"].values)
            if len(x) < 10:
                continue

        reg = linregress(x, y)
        t_crit = t_dist.ppf(0.975, len(x) - 2)
        ci_low = reg.slope - t_crit * reg.stderr
        ci_high = reg.slope + t_crit * reg.stderr

        color = provider_color(model_id)
        dash = model_dash[model_id]
        name = short_model(model_id)
        sig = sig_stars(reg.pvalue)

        slope_results.append({
            "model_id": model_id,
            "model": name,
            "slope": reg.slope,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p": reg.pvalue,
            "n": len(x),
        })

        # Scatter points (no legend entry)
        fig_scatter.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.08),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Regression line (legend entry with slope)
        x_line = np.array([x.min(), x.max()]) if fixed_effects else np.array([0.0, 1.0])
        y_line = reg.intercept + reg.slope * x_line
        fig_scatter.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(color=color, width=2.5, dash=dash),
            name=f"{name}  slope={reg.slope:.3f}{sig}",
            showlegend=True,
        ))

    if fixed_effects:
        x_title = "User Valence (within-proposition)"
        y_label = "Credence logit (within-prop.)" if use_logit else "Credence (within-prop.)"
        x_range, y_range = None, None
    else:
        x_title = "User Valence"
        y_label = "Credence (logit)" if use_logit else "Credence"
        x_range = [-0.05, 1.05]
        y_range = [0, 1] if not use_logit else None
    fig_scatter.update_layout(
        height=450,
        xaxis=dict(title=x_title, range=x_range),
        yaxis=dict(title=y_label, range=y_range),
        margin=dict(l=50, r=10, t=10, b=40),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Slope forest plot
    if slope_results:
        st.markdown("**Regression Slope Comparison**")
        fig_slopes = go.Figure()
        for res in slope_results:
            color = provider_color(res["model_id"])
            sig = sig_stars(res["p"])
            fig_slopes.add_trace(go.Scatter(
                x=[res["slope"]],
                y=[res["model"]],
                mode="markers",
                marker=dict(size=10, color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[res["ci_high"] - res["slope"]],
                    arrayminus=[res["slope"] - res["ci_low"]],
                    thickness=2,
                    width=6,
                    color=color,
                ),
                hovertemplate=(
                    f"slope = {res['slope']:.3f}{sig}<br>"
                    f"95% CI: [{res['ci_low']:.3f}, {res['ci_high']:.3f}]"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        scale_label = "logit" if use_logit else "credence"
        fig_slopes.update_layout(
            height=max(200, len(slope_results) * 50),
            margin=dict(l=10, r=10, t=10, b=55),
            xaxis=dict(title=f"Slope (Δ {scale_label} per unit valence)"),
            yaxis=dict(title=""),
            annotations=[
                dict(
                    text="Points = OLS slope. Whiskers = 95% CI.",
                    xref="paper", yref="paper",
                    x=0, y=-0.35,
                    xanchor="left", yanchor="top",
                    showarrow=False,
                    font=dict(size=10, color="#666"),
                )
            ],
        )
        st.plotly_chart(fig_slopes, use_container_width=True)

    # ── Section 2: Correlation Forest Plot ────────────────────────────────
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
        sel_cols = ["consensus_credence", "consensus_author_valence"]
        if fixed_effects:
            sel_cols.append("proposition")
        valid_data = model_data.select(sel_cols).drop_nulls().to_pandas()
        if len(valid_data) < 10:
            continue

        x = valid_data["consensus_author_valence"].values
        y = valid_data["consensus_credence"].values

        if fixed_effects:
            x, y = _demean_within_proposition(x, y, valid_data["proposition"].values)
            if len(x) < 10:
                continue

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

    # ── Section 3: Distribution Histograms ────────────────────────────────
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

    # ── Section 4: User Valence Distribution ──────────────────────────────
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
