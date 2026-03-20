"""Inspect tab — Deep-dive into individual propositions and samples."""

import statistics

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import streamlit as st

from credence.viz.data import load_calibration, load_china, load_monotonicity, load_negation
from credence.viz.formatting import is_chinese_model, natural_sort_key, provider, short_model, truncate

# Nationality-based colors for Known Divergence histograms
_MODEL_COLORS: dict[str, str] = {
    "openai": "#1F77B4",     # darker blue
    "anthropic": "#6BAED6",  # lighter blue
    "deepseek": "#E31A1C",   # darker red
    "moonshot": "#FC9272",   # lighter red/salmon
}


def render() -> None:
    """Render the inspect tab."""
    st.subheader("Inspect: Examine individual propositions and samples")
    st.caption(
        "Select a dataset and proposition to view per-sample details including "
        "prompts, model responses, and judge evaluations."
    )

    dataset_name = st.radio(
        "Dataset",
        ["Calibration", "Negation", "Monotonicity", "Known Divergence"],
        horizontal=True,
    )

    if dataset_name == "Negation":
        _render_negation_inspect()
    elif dataset_name == "Monotonicity":
        _render_monotonicity_inspect()
    else:
        _render_standard_inspect(dataset_name)


# ---------------------------------------------------------------------------
# Standard inspect (Calibration, Known Divergence)
# ---------------------------------------------------------------------------

def _render_standard_inspect(dataset_name):
    """Inspect flow for datasets with consensus_credence column."""
    loader = load_calibration if dataset_name == "Calibration" else load_china
    df = loader()
    if df.is_empty():
        st.info(f"No {dataset_name} data available.")
        return

    if "prompt_text" not in df.columns or "response_text" not in df.columns:
        st.warning("Re-export with the latest exporter to populate prompt/response text.")
        return

    # Filters
    df = _apply_model_filter(df)
    df = _apply_category_filter(df)

    is_known_div = dataset_name == "Known Divergence"
    prop_stats = _compute_standard_prop_stats(df, is_known_div)
    if prop_stats.is_empty():
        st.warning("No propositions found.")
        return

    props = prop_stats["proposition"].to_list()

    if is_known_div:
        w_meds = prop_stats["western_median"].to_list()
        c_meds = prop_stats["chinese_median"].to_list()
        prop_to_label = {}
        for p, w, c in zip(props, w_meds, c_meds):
            w_str = f"{w:.2f}" if w is not None else "N/A"
            c_str = f"{c:.2f}" if c is not None else "N/A"
            p_short = truncate(p, 55)
            prop_to_label[p] = f"W:{w_str} C:{c_str} — {p_short}"
    else:
        medians = prop_stats["median"].to_list()
        prop_to_label = {}
        for p, m in zip(props, medians):
            m_str = f"{m:.2f}" if m is not None else "N/A"
            p_short = truncate(p, 60)
            prop_to_label[p] = f"{m_str} — {p_short}"

    selected_prop = st.selectbox(
        "Proposition (by median credence)",
        props,
        format_func=lambda p: prop_to_label[p],
        key="inspect_prop",
    )

    prop_df = df.filter(pl.col("proposition") == selected_prop)
    _render_standard_header(selected_prop, prop_df, is_known_div)
    _render_standard_distribution_and_donuts(prop_df, is_known_div)

    st.divider()
    _render_standard_samples(prop_df)


def _compute_standard_prop_stats(df, is_known_div):
    """Compute per-proposition summary stats for standard datasets."""
    result = (
        df.group_by("proposition")
        .agg([
            pl.col("consensus_credence").drop_nulls().median().alias("median"),
            pl.len().alias("n_total"),
        ])
        .sort(["median", "proposition"], descending=[True, False])
    )

    if is_known_div and "target_model" in df.columns:
        all_models = df["target_model"].unique().to_list()
        chinese_models = [m for m in all_models if is_chinese_model(m)]
        western_models = [m for m in all_models if not is_chinese_model(m)]

        western_med = (
            df.filter(
                pl.col("target_model").is_in(western_models)
                & pl.col("consensus_credence").is_not_null()
            )
            .group_by("proposition")
            .agg(pl.col("consensus_credence").median().alias("western_median"))
        )
        chinese_med = (
            df.filter(
                pl.col("target_model").is_in(chinese_models)
                & pl.col("consensus_credence").is_not_null()
            )
            .group_by("proposition")
            .agg(pl.col("consensus_credence").median().alias("chinese_median"))
        )
        result = result.join(western_med, on="proposition", how="left")
        result = result.join(chinese_med, on="proposition", how="left")

    return result


def _render_standard_header(prop_text, prop_df, is_known_div):
    """Render proposition text with median credence(s)."""
    if is_known_div and "target_model" in prop_df.columns:
        all_models = prop_df["target_model"].unique().to_list()
        chinese_models = [m for m in all_models if is_chinese_model(m)]
        western_models = [m for m in all_models if not is_chinese_model(m)]

        w_vals = prop_df.filter(
            pl.col("target_model").is_in(western_models)
            & pl.col("consensus_credence").is_not_null()
        )["consensus_credence"].to_list()
        c_vals = prop_df.filter(
            pl.col("target_model").is_in(chinese_models)
            & pl.col("consensus_credence").is_not_null()
        )["consensus_credence"].to_list()
        w_med = statistics.median(w_vals) if w_vals else None
        c_med = statistics.median(c_vals) if c_vals else None
        w_str = f"{w_med:.2f}" if w_med is not None else "N/A"
        c_str = f"{c_med:.2f}" if c_med is not None else "N/A"
        st.markdown(f"**{prop_text}**  \nWestern median: `{w_str}` | Chinese median: `{c_str}`")
    else:
        consensus_vals = prop_df.filter(
            pl.col("consensus_credence").is_not_null()
        )["consensus_credence"].to_list()
        median_val = statistics.median(consensus_vals) if consensus_vals else None
        m_str = f"{median_val:.2f}" if median_val is not None else "N/A"
        st.markdown(f"**{prop_text}**  \nMedian credence: `{m_str}`")


def _standard_judge_breakdown(sub_df):
    """Compute (consensus, uninformative, disagreement) counts."""
    n = sub_df.height
    consensus = sub_df.filter(pl.col("consensus_credence").is_not_null()).height
    uninf = 0
    if "judge1_informative" in sub_df.columns:
        uninf = sub_df.filter(
            ~pl.col("judge1_informative") | ~pl.col("judge2_informative")
        ).height
    disagree = n - consensus - uninf
    return consensus, uninf, disagree


def _render_standard_distribution_and_donuts(prop_df, is_known_div):
    """Render histogram + donut chart(s) for standard datasets."""
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
                    marker_color=_MODEL_COLORS.get(provider(model), "#888888"),
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
        labels = ["Consensus", "Uninformative", "Disagreement"]
        colors = ["#2ecc71", "#e74c3c", "#f39c12"]

        if is_known_div and "target_model" in prop_df.columns:
            all_models = prop_df["target_model"].unique().to_list()
            chinese_models = [m for m in all_models if is_chinese_model(m)]
            western_models = [m for m in all_models if not is_chinese_model(m)]

            western_df = prop_df.filter(pl.col("target_model").is_in(western_models))
            chinese_df = prop_df.filter(pl.col("target_model").is_in(chinese_models))
            w_ok, w_uninf, w_dis = _standard_judge_breakdown(western_df)
            c_ok, c_uninf, c_dis = _standard_judge_breakdown(chinese_df)

            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "pie"}, {"type": "pie"}]],
                subplot_titles=["Western", "Chinese"],
            )
            fig.add_trace(go.Pie(
                labels=labels, values=[w_ok, w_uninf, w_dis],
                marker=dict(colors=colors),
                textinfo="percent", textposition="inside",
                hole=0.4, showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Pie(
                labels=labels, values=[c_ok, c_uninf, c_dis],
                marker=dict(colors=colors),
                textinfo="percent", textposition="inside",
                hole=0.4, showlegend=False,
            ), row=1, col=2)
            fig.update_layout(height=280, margin=dict(t=30, b=20))
        else:
            ok, uninf, dis = _standard_judge_breakdown(prop_df)
            fig = go.Figure(go.Pie(
                labels=labels, values=[ok, uninf, dis],
                marker=dict(colors=colors),
                textinfo="percent+label", textposition="inside",
                hole=0.4,
            ))
            fig.update_layout(height=280, margin=dict(t=20, b=20), showlegend=False)

        st.plotly_chart(fig, use_container_width=True)


def _render_standard_samples(prop_df):
    """Render expandable sample cards for standard datasets."""
    rows = prop_df.to_dicts()
    rows.sort(key=lambda s: (0, -(s.get("consensus_credence") or 0)) if s.get("consensus_credence") is not None else (1, 0))

    st.markdown(f"**Samples** ({len(rows)} total)")

    for s in rows:
        cc = s.get("consensus_credence")
        model_name = short_model(s.get("target_model", "unknown"))
        prompt_preview = str(s.get("prompt_text", ""))[:80].replace("\n", " ")

        j1_c = s.get("judge1_credence")
        j2_c = s.get("judge2_credence")
        j1_inf = s.get("judge1_informative", True)
        j2_inf = s.get("judge2_informative", True)
        j1_ref = s.get("judge1_refusal", False)
        j2_ref = s.get("judge2_refusal", False)

        header = f"**{model_name}** | "
        if cc is not None:
            header += f"{cc:.2f}"
        else:
            header += (
                f"No consensus "
                f"(J1={_judge_str(j1_c, j1_inf, j1_ref)}, "
                f"J2={_judge_str(j2_c, j2_inf, j2_ref)})"
            )
        header += f" | {prompt_preview}"

        with st.expander(header, expanded=False):
            st.markdown("**Prompt:**")
            st.code(str(s.get("prompt_text", ""))[:2000], language=None)
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


# ---------------------------------------------------------------------------
# Negation inspect
# ---------------------------------------------------------------------------

def _render_negation_inspect():
    """Inspect flow for negation data (p_consensus / notp_consensus columns)."""
    df = load_negation()
    if df.is_empty():
        st.info("No negation data available.")
        return

    if "prompt_text" not in df.columns or "response_text" not in df.columns:
        st.warning("Re-export with the latest exporter to populate prompt/response text.")
        return

    df = _apply_category_filter(df)

    # Compute per-pair stats
    if "pair_id" not in df.columns:
        st.warning("Negation data missing pair_id column.")
        return

    pair_stats = (
        df.group_by("pair_id")
        .agg([
            pl.col("proposition_p").first().alias("prop_p"),
            pl.col("proposition_not_p").first().alias("prop_notp"),
            pl.col("p_consensus").drop_nulls().median().alias("median_p"),
            pl.col("notp_consensus").drop_nulls().median().alias("median_notp"),
            pl.col("consistency_error").drop_nulls().median().alias("median_error"),
            pl.len().alias("n_total"),
        ])
        .sort(["median_error", "pair_id"], descending=[True, False])
    )

    if pair_stats.is_empty():
        st.warning("No negation pairs found.")
        return

    pair_ids = pair_stats["pair_id"].to_list()
    props_p = pair_stats["prop_p"].to_list()
    med_p = pair_stats["median_p"].to_list()
    med_notp = pair_stats["median_notp"].to_list()
    med_err = pair_stats["median_error"].to_list()

    pair_to_label = {}
    for pid, pp, mp, mnp, me in zip(pair_ids, props_p, med_p, med_notp, med_err):
        p_str = f"{mp:.2f}" if mp is not None else "?"
        np_str = f"{mnp:.2f}" if mnp is not None else "?"
        e_str = f"{me:.3f}" if me is not None else "?"
        p_short = truncate(pp, 45)
        pair_to_label[pid] = f"err:{e_str} P:{p_str} ¬P:{np_str} — {p_short}"

    selected_pair = st.selectbox(
        "Negation pair (by median error)",
        pair_ids,
        format_func=lambda pid: pair_to_label[pid],
        key="inspect_neg_pair",
    )

    pair_df = df.filter(pl.col("pair_id") == selected_pair)
    first = pair_df.row(0, named=True)
    prop_p = first.get("proposition_p", "")
    prop_notp = first.get("proposition_not_p", "")

    # Header
    p_vals = pair_df.filter(pl.col("p_consensus").is_not_null())["p_consensus"].to_list()
    np_vals = pair_df.filter(pl.col("notp_consensus").is_not_null())["notp_consensus"].to_list()
    p_med = statistics.median(p_vals) if p_vals else None
    np_med = statistics.median(np_vals) if np_vals else None
    err_vals = pair_df.filter(pl.col("consistency_error").is_not_null())["consistency_error"].to_list()
    err_med = statistics.median(err_vals) if err_vals else None

    st.markdown(f"**P:** {prop_p}")
    st.markdown(f"**¬P:** {prop_notp}")
    p_str = f"{p_med:.2f}" if p_med is not None else "N/A"
    np_str = f"{np_med:.2f}" if np_med is not None else "N/A"
    err_str = f"{err_med:.3f}" if err_med is not None else "N/A"
    st.markdown(f"Median P: `{p_str}` | Median ¬P: `{np_str}` | Consistency error: `{err_str}`")

    # Histogram + donut
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        if p_vals:
            fig.add_trace(go.Histogram(
                x=p_vals, name="P", marker_color="#3498db", opacity=0.7,
                xbins=dict(start=0, end=1, size=0.05),
            ))
        if np_vals:
            fig.add_trace(go.Histogram(
                x=np_vals, name="¬P", marker_color="#e67e22", opacity=0.7,
                xbins=dict(start=0, end=1, size=0.05),
            ))
        fig.update_layout(
            barmode="overlay",
            xaxis=dict(title="Consensus Credence", range=[0, 1]),
            yaxis=dict(title="Count"),
            height=280, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        # Donut for P-side consensus breakdown
        n = pair_df.height
        p_consensus_n = pair_df.filter(pl.col("p_consensus").is_not_null()).height
        p_uninf = pair_df.filter(
            pl.col("p_judge1_credence").is_null() | pl.col("p_judge2_credence").is_null()
        ).height
        p_disagree = n - p_consensus_n - p_uninf

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "pie"}]],
            subplot_titles=["P side", "¬P side"],
        )
        labels = ["Consensus", "Uninformative", "Disagreement"]
        colors = ["#2ecc71", "#e74c3c", "#f39c12"]

        fig.add_trace(go.Pie(
            labels=labels, values=[p_consensus_n, p_uninf, p_disagree],
            marker=dict(colors=colors),
            textinfo="percent", textposition="inside",
            hole=0.4, showlegend=False,
        ), row=1, col=1)

        np_consensus_n = pair_df.filter(pl.col("notp_consensus").is_not_null()).height
        np_uninf = pair_df.filter(
            pl.col("notp_judge1_credence").is_null() | pl.col("notp_judge2_credence").is_null()
        ).height
        np_disagree = n - np_consensus_n - np_uninf

        fig.add_trace(go.Pie(
            labels=labels, values=[np_consensus_n, np_uninf, np_disagree],
            marker=dict(colors=colors),
            textinfo="percent", textposition="inside",
            hole=0.4, showlegend=False,
        ), row=1, col=2)
        fig.update_layout(height=280, margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Sample cards
    rows = pair_df.to_dicts()
    rows.sort(key=lambda s: abs(s.get("consistency_error") or 999))
    st.markdown(f"**Samples** ({len(rows)} total)")

    for s in rows:
        model_name = short_model(s.get("target_model", "unknown"))
        p_c = s.get("p_consensus")
        np_c = s.get("notp_consensus")
        err = s.get("consistency_error")
        prompt_preview = str(s.get("prompt_text", ""))[:80].replace("\n", " ")

        p_str = f"P:{p_c:.2f}" if p_c is not None else "P:—"
        np_str = f"¬P:{np_c:.2f}" if np_c is not None else "¬P:—"
        err_str = f"err:{err:.3f}" if err is not None else "err:—"
        header = f"**{model_name}** | {p_str} {np_str} {err_str} | {prompt_preview}"

        with st.expander(header, expanded=False):
            st.markdown("**Prompt:**")
            st.code(str(s.get("prompt_text", ""))[:2000], language=None)
            st.markdown("**Response:**")
            st.code(str(s.get("response_text", ""))[:3000], language=None)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**P-side judges:**")
                pj1 = s.get("p_judge1_credence")
                pj2 = s.get("p_judge2_credence")
                st.markdown(
                    f"J1: `{pj1:.2f}`" if pj1 is not None else "J1: non-informative"
                )
                st.markdown(
                    f"J2: `{pj2:.2f}`" if pj2 is not None else "J2: non-informative"
                )
                st.markdown(f"Consensus: `{p_c:.2f}`" if p_c is not None else "Consensus: —")
            with col2:
                st.markdown("**¬P-side judges:**")
                npj1 = s.get("notp_judge1_credence")
                npj2 = s.get("notp_judge2_credence")
                st.markdown(
                    f"J1: `{npj1:.2f}`" if npj1 is not None else "J1: non-informative"
                )
                st.markdown(
                    f"J2: `{npj2:.2f}`" if npj2 is not None else "J2: non-informative"
                )
                st.markdown(f"Consensus: `{np_c:.2f}`" if np_c is not None else "Consensus: —")


# ---------------------------------------------------------------------------
# Monotonicity inspect
# ---------------------------------------------------------------------------

def _render_monotonicity_inspect():
    """Inspect flow for monotonicity data (credences list, series_id)."""
    df = load_monotonicity()
    if df.is_empty():
        st.info("No monotonicity data available.")
        return

    if "prompt_text" not in df.columns or "response_text" not in df.columns:
        st.warning("Re-export with the latest exporter to populate prompt/response text.")
        return

    if "series_id" not in df.columns:
        st.warning("Monotonicity data missing series_id column.")
        return

    series_ids = sorted(df["series_id"].unique().to_list(), key=natural_sort_key)

    # Build labels
    series_to_label = {}
    for sid in series_ids:
        sdf = df.filter(pl.col("series_id") == sid)
        first = sdf.row(0, named=True)
        props = first.get("propositions", [])
        passes = sdf.filter(pl.col("is_monotonic") == True).height
        total = sdf.filter(pl.col("is_monotonic").is_not_null()).height
        rate_str = f"{passes}/{total}" if total > 0 else "—"
        prop_short = truncate(props[0], 40) if props else "?"
        series_to_label[sid] = f"[{rate_str}] {prop_short}"

    selected_series = st.selectbox(
        "Series (by pass rate)",
        series_ids,
        format_func=lambda s: series_to_label[s],
        key="inspect_mono_series",
    )

    series_df = df.filter(pl.col("series_id") == selected_series)
    first = series_df.row(0, named=True)
    props = first.get("propositions", [])
    n_props = len(props)

    # Header
    for i, p in enumerate(props):
        st.markdown(f"**P{i+1}:** {p}")

    passes = series_df.filter(pl.col("is_monotonic") == True).height
    total = series_df.filter(pl.col("is_monotonic").is_not_null()).height
    st.markdown(f"Monotonicity: `{passes}/{total}` samples pass")

    # Collect per-position credences
    per_pos: list[list[float]] = [[] for _ in range(n_props)]
    for row in series_df.iter_rows(named=True):
        creds = row.get("credences")
        if not creds:
            continue
        for i, c in enumerate(creds):
            if i < n_props and c is not None:
                per_pos[i].append(c)

    # Box plot
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        position_labels = [f"P{i+1}" for i in range(n_props)]
        for i in range(n_props):
            fig.add_trace(go.Box(
                y=per_pos[i],
                name=position_labels[i],
                boxpoints="all", jitter=0.3, pointpos=0,
                marker=dict(opacity=0.4, size=4),
            ))
        medians = [statistics.median(c) if c else None for c in per_pos]
        valid_meds = [(i, m) for i, m in enumerate(medians) if m is not None]
        if len(valid_meds) > 1:
            fig.add_trace(go.Scatter(
                x=[position_labels[i] for i, _ in valid_meds],
                y=[m for _, m in valid_meds],
                mode="lines+markers",
                line=dict(color="gray", width=2),
                marker=dict(size=8, symbol="diamond"),
                showlegend=False,
            ))
        fig.update_layout(
            yaxis=dict(title="Credence", range=[0, 1]),
            height=300, margin=dict(t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        # Donut: consensus / uninformative per position (aggregated)
        if "judgments" in series_df.columns:
            total_ok = total_uninf = total_disagree = 0
            for row in series_df.iter_rows(named=True):
                judgments = row.get("judgments")
                creds = row.get("credences")
                if not judgments:
                    continue
                for pi, j in enumerate(judgments):
                    if pi >= n_props:
                        break
                    has_consensus = creds and pi < len(creds) and creds[pi] is not None
                    j1_inf = j.get("j1", {}).get("informative", True)
                    j2_inf = j.get("j2", {}).get("informative", True)
                    if has_consensus:
                        total_ok += 1
                    elif not j1_inf or not j2_inf:
                        total_uninf += 1
                    else:
                        total_disagree += 1

            labels = ["Consensus", "Uninformative", "Disagreement"]
            colors = ["#2ecc71", "#e74c3c", "#f39c12"]
            fig = go.Figure(go.Pie(
                labels=labels, values=[total_ok, total_uninf, total_disagree],
                marker=dict(colors=colors),
                textinfo="percent+label", textposition="inside",
                hole=0.4,
            ))
            fig.update_layout(height=300, margin=dict(t=20, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Sample cards
    rows = series_df.to_dicts()
    st.markdown(f"**Samples** ({len(rows)} total)")

    for s in rows:
        model_name = short_model(s.get("target_model", "unknown"))
        creds = s.get("credences", [])
        is_mono = s.get("is_monotonic")
        prompt_preview = str(s.get("prompt_text", ""))[:80].replace("\n", " ")

        cred_strs = [f"{c:.2f}" if c is not None else "—" for c in (creds or [])]
        mono_str = "PASS" if is_mono else "FAIL" if is_mono is not None else "—"
        header = f"**{model_name}** | [{', '.join(cred_strs)}] {mono_str} | {prompt_preview}"

        with st.expander(header, expanded=False):
            st.markdown("**Prompt:**")
            st.code(str(s.get("prompt_text", ""))[:2000], language=None)
            st.markdown("**Response:**")
            st.code(str(s.get("response_text", ""))[:3000], language=None)

            if creds:
                for i, c in enumerate(creds):
                    prop_text = props[i] if i < len(props) else f"Position {i+1}"
                    c_str = f"`{c:.2f}`" if c is not None else "—"
                    st.markdown(f"**P{i+1}**: {c_str} — {prop_text}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_model_filter(df):
    """Apply optional multi-model filter widget."""
    if "target_model" in df.columns:
        all_models = sorted(df["target_model"].unique().to_list())
        if len(all_models) > 1:
            selected_models = st.multiselect(
                "Filter models", all_models, default=all_models, key="inspect_models",
            )
            if selected_models:
                df = df.filter(pl.col("target_model").is_in(selected_models))
    return df


def _apply_category_filter(df):
    """Apply optional category filter widget."""
    if "category" in df.columns:
        categories = sorted(df["category"].unique().to_list())
        if len(categories) > 1:
            selected_cat = st.selectbox(
                "Category", ["All"] + categories, key="inspect_cat",
            )
            if selected_cat != "All":
                df = df.filter(pl.col("category") == selected_cat)
    return df


def _judge_str(credence, informative, refusal):
    """Format a single judge's result for the expander header."""
    if refusal:
        return "refusal"
    if not informative:
        return "non-inf"
    if credence is not None:
        return f"{credence:.2f}"
    return "N/A"
