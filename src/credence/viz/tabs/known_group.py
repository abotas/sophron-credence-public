"""V5 Known Divergence — China/Western Comparison."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import streamlit as st

from credence.viz.constants import MODEL_GROUP_COLORS
from credence.viz.data import load_china
from credence.viz.formatting import is_chinese_model, short_model, truncate
from credence.viz.stats import bootstrap_mean_ci, wilcoxon_one_sided

# Dumbbell chart colors — by nationality
_COLOR_WESTERN = MODEL_GROUP_COLORS["western"]
_COLOR_CHINESE = MODEL_GROUP_COLORS["chinese"]


def render() -> None:
    """Render the Known-Group validity tab."""
    st.subheader("V5 Known Divergence: Do Chinese/Western models diverge on sensitive topics?")
    st.caption(
        "We check whether our methodology can detect a known cross-model difference. "
        "We compare Chinese models (DeepSeek, Kimi) to Western models (Claude, GPT-5-mini) "
        "on 50 politically sensitive propositions with pre-specified expected directions. "
        "4 target models, 32 prompts per proposition."
    )

    df = load_china()
    if df.is_empty():
        st.info("No Known Divergence data. Export results to `china.jsonl`.")
        return

    # Classify models
    all_models = sorted(df["target_model"].unique().to_list())
    western_models = [m for m in all_models if not is_chinese_model(m)]
    chinese_models = [m for m in all_models if is_chinese_model(m)]

    if not western_models or not chinese_models:
        st.warning("Need both Western and Chinese models for comparison.")
        return

    # Compute per-proposition shifts
    results = _compute_proposition_results(df, western_models, chinese_models)
    plot_results = [r for r in results if r["western"] is not None and r["chinese"] is not None]

    if not plot_results:
        st.warning("No propositions with data from both model groups.")
        return

    # 1. Headline metrics
    _render_metrics(results)

    # 2. Dumbbell chart (the main visualization)
    st.subheader("Per-Proposition Divergence")
    _render_dumbbell_chart(plot_results)

    # Order: Western first, then Chinese
    ordered_models = western_models + chinese_models

    # 3. Judge & consensus rates per model
    st.subheader("Judge & Consensus Rates")
    _render_judge_donuts(df, ordered_models)

    # 4. Refusal rates
    st.subheader("Refusal Rates")
    _render_refusal_table(df, ordered_models)

    # 5. Per-proposition detail
    with st.expander("Per-Proposition Detail"):
        _render_proposition_detail(plot_results)


def _compute_proposition_results(
    df: pl.DataFrame,
    western_models: list[str],
    chinese_models: list[str],
) -> list[dict]:
    """Compute per-proposition mean credence by model group and directional shift."""
    # Median consensus credence per (proposition, target_model)
    prop_model = (
        df.filter(pl.col("consensus_credence").is_not_null())
        .group_by(["proposition", "target_model"])
        .agg(pl.col("consensus_credence").mean().alias("mean_cred"))
    )

    # Determine direction column
    direction_col = None
    for col_name in ["china_sensitivity_direction", "direction", "expected_direction"]:
        if col_name in df.columns:
            direction_col = col_name
            break

    results = []
    for prop in sorted(df["proposition"].unique().to_list()):
        prop_df = prop_model.filter(pl.col("proposition") == prop)

        western_creds = prop_df.filter(
            pl.col("target_model").is_in(western_models)
        )["mean_cred"].to_list()
        chinese_creds = prop_df.filter(
            pl.col("target_model").is_in(chinese_models)
        )["mean_cred"].to_list()

        if not western_creds or not chinese_creds:
            results.append({
                "proposition": prop, "direction": None,
                "western": None, "chinese": None,
                "shift": None, "signed_shift": None, "correct": None,
            })
            continue

        western = float(np.mean(western_creds))
        chinese = float(np.mean(chinese_creds))
        shift = chinese - western

        # Get direction
        direction = None
        if direction_col:
            dir_vals = df.filter(pl.col("proposition") == prop)[direction_col].unique().to_list()
            if dir_vals:
                raw = str(dir_vals[0]).lower().strip()
                # Handle both bool-string ("true"/"false") and label ("pro_china"/"anti_china")
                direction = raw in ("true", "pro_china")

        if direction is not None:
            signed_shift = shift if direction else -shift
            correct = signed_shift > 0
        else:
            signed_shift = None
            correct = None

        results.append({
            "proposition": prop,
            "direction": direction,
            "western": western,
            "chinese": chinese,
            "shift": shift,
            "signed_shift": signed_shift,
            "correct": correct,
        })

    return results


def _render_metrics(results: list[dict]) -> None:
    """Render headline KPI metrics."""
    valid_shifts = [r["signed_shift"] for r in results if r["signed_shift"] is not None]
    valid_results = [r for r in results if r["correct"] is not None]
    n_correct = sum(1 for r in valid_results if r["correct"])
    n_total = len(valid_results)

    mean_shift = sum(valid_shifts) / len(valid_shifts) if valid_shifts else None
    nonzero = [s for s in valid_shifts if s != 0]

    p_str = "N/A"
    if len(nonzero) >= 5:
        _, p_val = wilcoxon_one_sided(nonzero)
        p_str = f"{p_val:.4f}"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Mean Signed Shift",
            f"{mean_shift:+.3f}" if mean_shift is not None else "N/A",
        )
    with col2:
        st.metric("Wilcoxon p-value", p_str)
    with col3:
        st.metric(
            "Directional Accuracy",
            f"{n_correct / n_total:.0%}" if n_total > 0 else "N/A",
            help=f"{n_correct}/{n_total}",
        )


def _render_dumbbell_chart(plot_results: list[dict]) -> None:
    """Render the dumbbell chart showing Western vs Chinese credences per proposition."""
    # Sort by signed_shift ascending (biggest correct at top, biggest wrong at bottom)
    plot_results = sorted(plot_results, key=lambda r: r["signed_shift"] or 0)

    fig = go.Figure()

    for r in plot_results:
        prop_short = truncate(r["proposition"], 40)
        prop_full = r["proposition"]
        shift_val = r["shift"]

        # Line connecting Western to Chinese (green=expected, red=unexpected)
        line_color = "#4daf4a" if r["correct"] else "#e41a1c"
        fig.add_trace(go.Scatter(
            x=[r["western"], r["chinese"]],
            y=[prop_short, prop_short],
            mode="lines",
            line=dict(color=line_color, width=2, dash="solid"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Western dot (blue circle)
        fig.add_trace(go.Scatter(
            x=[r["western"]],
            y=[prop_short],
            mode="markers",
            marker=dict(color=_COLOR_WESTERN, size=10, symbol="circle"),
            showlegend=False,
            hovertemplate=(
                f"<b>{prop_full}</b><br>"
                f"Western: {r['western']:.3f}<br>"
                f"Shift: {shift_val:+.3f}<extra></extra>"
            ),
        ))

        # Chinese dot (red diamond)
        fig.add_trace(go.Scatter(
            x=[r["chinese"]],
            y=[prop_short],
            mode="markers",
            marker=dict(color=_COLOR_CHINESE, size=10, symbol="diamond"),
            showlegend=False,
            hovertemplate=(
                f"<b>{prop_full}</b><br>"
                f"Chinese: {r['chinese']:.3f}<br>"
                f"Shift: {shift_val:+.3f}<extra></extra>"
            ),
        ))

    # Legend entries
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=_COLOR_WESTERN, size=10, symbol="circle"),
        name="Western",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=_COLOR_CHINESE, size=10, symbol="diamond"),
        name="Chinese",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="#4daf4a", width=2, dash="solid"),
        name="Expected direction",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="#e41a1c", width=2, dash="solid"),
        name="Unexpected direction",
    ))

    fig.update_layout(
        height=max(300, 25 * len(plot_results)),
        xaxis=dict(title="Credence", range=[0, 1]),
        yaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_judge_donuts(df: pl.DataFrame, models: list[str]) -> None:
    """Render one donut chart per target model showing consensus/uninformative/disagreement."""
    if "judge1_informative" not in df.columns:
        return

    n_models = len(models)
    fig = make_subplots(
        rows=1, cols=n_models,
        specs=[[{"type": "pie"}] * n_models],
        subplot_titles=[short_model(m) for m in models],
    )

    for i, model in enumerate(models):
        mdf = df.filter(pl.col("target_model") == model)
        n = mdf.height
        if n == 0:
            continue

        consensus = mdf.filter(pl.col("consensus_credence").is_not_null()).height
        uninf = mdf.filter(
            ~pl.col("judge1_informative") | ~pl.col("judge2_informative")
        ).height
        disagree = n - consensus - uninf

        fig.add_trace(go.Pie(
            labels=["Consensus", "Uninformative", "Disagreement"],
            values=[consensus, uninf, disagree],
            marker=dict(colors=["#2ecc71", "#e74c3c", "#f39c12"]),
            textinfo="percent",
            textposition="inside",
            hole=0.4,
        ), row=1, col=i + 1)

    fig.update_layout(height=250, margin=dict(t=40, b=20), showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)


def _render_refusal_table(df: pl.DataFrame, models: list[str]) -> None:
    """Render refusal rate table per model."""
    rows = []
    for model in models:
        model_df = df.filter(pl.col("target_model") == model)
        total = model_df.height
        refusals = 0
        if "judge1_refusal" in model_df.columns:
            refusals = model_df.filter(
                pl.col("judge1_refusal") | pl.col("judge2_refusal")
            ).height

        rows.append({
            "Model": short_model(model),
            "Group": "Chinese" if is_chinese_model(model) else "Western",
            "Refusals": refusals,
            "Total": total,
            "Refusal Rate": f"{refusals / total:.1%}" if total > 0 else "—",
        })

    st.dataframe(pl.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_proposition_detail(plot_results: list[dict]) -> None:
    """Render per-proposition detail table."""
    rows = []
    for r in sorted(plot_results, key=lambda x: x["signed_shift"] or 0, reverse=True):
        rows.append({
            "Proposition": r["proposition"],
            "Direction": "China Higher" if r["direction"] else "West Higher",
            "Western": f"{r['western']:.3f}" if r["western"] is not None else "—",
            "Chinese": f"{r['chinese']:.3f}" if r["chinese"] is not None else "—",
            "Shift": f"{r['shift']:+.3f}" if r["shift"] is not None else "—",
            "Signed Shift": f"{r['signed_shift']:+.3f}" if r["signed_shift"] is not None else "—",
            "Correct": "Yes" if r["correct"] else "No" if r["correct"] is not None else "—",
        })

    st.dataframe(pl.DataFrame(rows), use_container_width=True, hide_index=True)
