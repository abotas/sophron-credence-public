#!/usr/bin/env python3
"""Human credence judgment app.

A Streamlit app where human judges provide credence scores for
interaction logs from the exploration pipeline, mirroring what
LLM judges do automatically.

Usage:
    uvr streamlit run src/credence/viz/human_judge.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import streamlit as st

from credence.core.export import RESULTS_DIR

# ── Paths ────────────────────────────────────────────────────────────────────

JUDGMENTS_DIR = Path(".data/human_judgments/credence_judgments")

# ── Data loading ─────────────────────────────────────────────────────────────


@st.cache_data
def load_exploration_items() -> pl.DataFrame:
    """Load exploration results that have existing AI judgments."""
    exploration_dir = RESULTS_DIR / "exploration"
    if not exploration_dir.exists():
        return pl.DataFrame()
    files = sorted(exploration_dir.glob("*.parquet"))
    if not files:
        return pl.DataFrame()

    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")

    # Only items with existing AI judgments
    if "score_value" in df.columns:
        df = df.filter(pl.col("score_value").is_not_null())

    keep = [
        c
        for c in [
            "sample_id",
            "target_model",
            "proposition_id",
            "proposition",
            "prompt_text",
            "response_text",
            "domain",
            "score_value",
        ]
        if c in df.columns
    ]
    # Sort deterministically first so new samples append to the end,
    # then shuffle with a fixed seed for stable cross-user ordering.
    df = df.select(keep).sort(["sample_id", "target_model"])
    return df.sample(fraction=1.0, seed=42, shuffle=True)


def load_user_judgments(username: str) -> set[tuple[str, str]]:
    """Load existing judgment keys for a user.

    Returns set of (sample_id, target_model) already judged.
    """
    path = JUDGMENTS_DIR / f"{username}.jsonl"
    if not path.exists():
        return set()
    judged: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            judged.add((record["sample_id"], record["target_model"]))
    return judged


def load_all_judgments() -> pl.DataFrame:
    """Load all human judgments from all users into a single DataFrame."""
    if not JUDGMENTS_DIR.exists():
        return pl.DataFrame()
    files = sorted(JUDGMENTS_DIR.glob("*.jsonl"))
    if not files:
        return pl.DataFrame()

    records: list[dict] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    if not records:
        return pl.DataFrame()

    # Deduplicate: keep last judgment per (username, sample_id, target_model)
    df = pl.DataFrame(records)
    df = df.unique(subset=["username", "sample_id", "target_model"], keep="last")
    return df


def save_judgment(username: str, record: dict) -> None:
    """Append a single judgment record to the user's JSONL file."""
    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = JUDGMENTS_DIR / f"{username}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Tab: Judge ───────────────────────────────────────────────────────────────


def tab_judge(username: str, items: pl.DataFrame) -> None:
    """Judgment entry tab."""
    judged_keys = load_user_judgments(username)

    # Track skipped items in session state (not persisted)
    if "skipped_keys" not in st.session_state:
        st.session_state.skipped_keys = set()

    all_rows = items.to_dicts()
    unjudged = [
        row for row in all_rows
        if (row["sample_id"], row["target_model"]) not in judged_keys
        and (row["sample_id"], row["target_model"]) not in st.session_state.skipped_keys
    ]

    total = len(all_rows)
    done = total - len(unjudged)

    # Sidebar: progress
    st.sidebar.divider()
    st.sidebar.metric("Progress", f"{done} / {total}")
    st.sidebar.progress(done / total if total > 0 else 0.0)

    if not unjudged:
        st.success(f"All {total} items judged. Thank you, {username}!")
        return

    st.sidebar.caption(f"{len(unjudged)} remaining")

    # Display current item
    item = unjudged[0]

    st.subheader("Proposition")
    st.markdown(f"**{item['proposition']}**")

    parts = []
    if item.get("domain"):
        parts.append(f"Domain: {item['domain']}")
    parts.append(f"Model: {item['target_model']}")
    st.caption("  |  ".join(parts))

    col_prompt, col_response = st.columns(2)
    with col_prompt:
        st.markdown("**Prompt**")
        st.text_area(
            "prompt_display",
            value=item.get("prompt_text") or "",
            height=350,
            disabled=True,
            label_visibility="collapsed",
        )
    with col_response:
        st.markdown("**Model Response**")
        st.text_area(
            "response_display",
            value=item.get("response_text") or "",
            height=350,
            disabled=True,
            label_visibility="collapsed",
        )

    # Judgment form
    st.divider()
    st.subheader("Your Judgment")

    with st.form("judgment_form"):
        refusal = st.checkbox(
            "Refusal",
            help="The model explicitly refused to engage with the proposition.",
        )

        informative = st.checkbox(
            "Informative",
            value=True,
            help="The response contains enough signal to estimate a credence.",
        )
        # Enforce mutual exclusion: refusal implies not informative
        if refusal:
            informative = False
            st.caption("Refusal checked — informative is forced to False.")

        credence: float | None = None
        if informative:
            credence = st.slider(
                "Credence (0.0 = certainly false, 1.0 = certainly true)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )

        explanation = st.text_area(
            "Explanation (optional)",
            placeholder="Brief reasoning for your judgment...",
            height=100,
        )

        col_submit, col_skip, _ = st.columns([1, 1, 3])
        with col_submit:
            submitted = st.form_submit_button("Submit Judgment", type="primary")
        with col_skip:
            skipped = st.form_submit_button("Skip")

    if skipped:
        st.session_state.skipped_keys.add((item["sample_id"], item["target_model"]))
        st.rerun()

    if submitted:
        record = {
            "username": username,
            "sample_id": item["sample_id"],
            "target_model": item["target_model"],
            "proposition_id": item.get("proposition_id"),
            "proposition": item["proposition"],
            "refusal": refusal,
            "informative": informative,
            "credence": credence,
            "explanation": explanation or None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_judgment(username, record)
        st.rerun()


# ── Tab: Results ─────────────────────────────────────────────────────────────


def tab_results(exploration: pl.DataFrame) -> None:
    """Results viewer tab showing all human judgments."""
    judgments = load_all_judgments()
    if judgments.is_empty():
        st.info("No human judgments recorded yet.")
        return

    # Filters
    all_users = sorted(judgments["username"].unique().to_list())
    selected_users = st.multiselect(
        "Filter by user",
        options=all_users,
        default=all_users,
        key="results_users",
    )
    if selected_users:
        judgments = judgments.filter(pl.col("username").is_in(selected_users))

    if judgments.is_empty():
        st.warning("No judgments match the selected filters.")
        return

    # Summary metrics
    n_judgments = len(judgments)
    n_informative = judgments.filter(pl.col("informative")).height
    n_refusals = judgments.filter(pl.col("refusal")).height
    mean_credence = (
        judgments.filter(pl.col("credence").is_not_null())["credence"].mean()
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total judgments", n_judgments)
    col2.metric("Informative", n_informative)
    col3.metric("Refusals", n_refusals)
    col4.metric("Mean credence", f"{mean_credence:.2f}" if mean_credence is not None else "N/A")

    # Join with exploration data to get AI scores for comparison
    if not exploration.is_empty() and "score_value" in exploration.columns:
        merged = judgments.join(
            exploration.select(["sample_id", "target_model", "score_value", "domain"]).unique(
                subset=["sample_id", "target_model"]
            ),
            on=["sample_id", "target_model"],
            how="left",
        )
    else:
        merged = judgments

    st.divider()

    # Human vs AI comparison (if AI scores available)
    if "score_value" in merged.columns:
        comparable = merged.filter(
            pl.col("credence").is_not_null() & pl.col("score_value").is_not_null()
        )
        if not comparable.is_empty():
            st.subheader("Human vs AI Credence")

            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=comparable["score_value"].to_list(),
                y=comparable["credence"].to_list(),
                mode="markers",
                marker=dict(size=5, opacity=0.5),
                text=[
                    f"{r['username']}<br>{r['proposition'][:80]}..."
                    for r in comparable.to_dicts()
                ],
                hoverinfo="text",
            ))
            fig.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1,
                line=dict(color="grey", dash="dash"),
            )
            fig.update_layout(
                xaxis_title="AI Credence (consensus)",
                yaxis_title="Human Credence",
                height=500,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Agreement stats
            diff = (comparable["credence"] - comparable["score_value"]).abs()
            mae = diff.mean()
            within_02 = (diff <= 0.2).sum() / len(diff) * 100
            st.caption(
                f"MAE: {mae:.3f}  |  "
                f"Agreement within 0.2: {within_02:.1f}%  |  "
                f"n = {len(comparable)}"
            )

    # Per-user summary
    st.subheader("Per-user summary")
    user_summary = (
        judgments.group_by("username")
        .agg(
            pl.len().alias("n_judgments"),
            pl.col("informative").sum().alias("n_informative"),
            pl.col("refusal").sum().alias("n_refusals"),
            pl.col("credence").mean().alias("mean_credence"),
        )
        .sort("username")
    )
    st.dataframe(user_summary, use_container_width=True, hide_index=True)

    # Full judgment table
    st.subheader("All judgments")
    display_cols = [
        c
        for c in [
            "username",
            "proposition",
            "target_model",
            "refusal",
            "informative",
            "credence",
            "explanation",
            "timestamp",
        ]
        if c in merged.columns
    ]
    if "score_value" in merged.columns:
        display_cols.insert(display_cols.index("credence") + 1, "score_value")
    if "domain" in merged.columns:
        display_cols.insert(2, "domain")

    st.dataframe(
        merged.select(display_cols).sort("timestamp", descending=True),
        use_container_width=True,
        hide_index=True,
        height=500,
    )


# ── App ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="Human Credence Judge", layout="wide")
    st.title("Human Credence Judge")

    tab = st.sidebar.radio("Mode", ["Judge", "Results"], key="hj_tab")

    # ── Load exploration data (shared) ────────────────────────────────────
    exploration = load_exploration_items()

    if tab == "Results":
        tab_results(exploration)
        return

    # ── Judge tab requires username ───────────────────────────────────────
    username = st.sidebar.text_input("Username", key="hj_username")
    if not username:
        st.info("Enter your username in the sidebar to begin judging.")
        return
    username = username.strip().lower().replace(" ", "_")

    if exploration.is_empty():
        st.error("No exploration data found. Run the exploration pipeline first.")
        return

    # Sidebar: domain filter
    all_domains = sorted(exploration["domain"].unique().to_list()) if "domain" in exploration.columns else []
    selected_domains = st.sidebar.multiselect(
        "Filter by domain",
        options=all_domains,
        default=all_domains,
        key="hj_domains",
    )
    items = exploration
    if selected_domains and "domain" in items.columns:
        items = items.filter(pl.col("domain").is_in(selected_domains))

    if items.is_empty():
        st.warning("No items match the selected domains.")
        return

    tab_judge(username, items)


if __name__ == "__main__":
    main()
