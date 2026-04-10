#!/usr/bin/env python3
"""Human credence judgment app — round 4 edition.

A Streamlit app where human judges provide credence scores for
(proposition, prompt, response) samples drawn from the exploration
pipeline, mirroring what the LLM credence judges do automatically.

Data sources
------------
- ``human_validation/samples_for_human_validation.jsonl.gz`` — the main
  pool, pre-built via ``build_validation_samples.py``. Each row has a
  deterministic UUID5 ``item_id``.
- ``human_validation/attention_checks.jsonl`` — curated attention checks
  that get interleaved into the judging stream at a fixed cadence.

Judgments are appended to ``human_validation/judgments/<username>.jsonl``.

Usage:
    uvr streamlit run human_validation/human_judge_app.py
"""

from __future__ import annotations

import gzip
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import streamlit as st


# ── Paths ────────────────────────────────────────────────────────────────────

_HV_DIR = Path(__file__).resolve().parent
SAMPLES_PATH = _HV_DIR / "samples_for_human_validation_1k.jsonl.gz"
ATTENTION_CHECKS_PATH = _HV_DIR / "attention_checks.jsonl"
JUDGMENTS_DIR = _HV_DIR / "judgments"

# One attention check every ATTN_CHECK_EVERY real items.
ATTN_CHECK_EVERY = 15


# ── Data loading ─────────────────────────────────────────────────────────────


def _is_attention_check(item_id: str) -> bool:
    """Attention checks are identified by their item_id prefix."""
    return item_id.startswith("attn__")


@st.cache_data
def load_samples() -> list[dict]:
    """Load the main sample pool from the gzipped JSONL."""
    if not SAMPLES_PATH.exists():
        return []
    with gzip.open(SAMPLES_PATH, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@st.cache_data
def load_attention_checks() -> list[dict]:
    """Load attention checks.

    Attention checks share the main samples schema (same field names),
    with item_id prefixed by "attn__" and an extra ``expected_credence``
    field for scoring the labeler.
    """
    if not ATTENTION_CHECKS_PATH.exists():
        return []
    rows = []
    with open(ATTENTION_CHECKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_display_order(username: str, samples: list[dict], attn_checks: list[dict]) -> list[dict]:
    """Return a deterministic shuffled + attention-check-interleaved order.

    Seeded on the username so each labeler gets a stable order and the
    same attention checks in the same positions.
    """
    rng = random.Random(f"human_judge_app::{username}")

    real = list(samples)
    rng.shuffle(real)

    attn = list(attn_checks)
    rng.shuffle(attn)

    if not attn:
        return real

    # Interleave: after every ATTN_CHECK_EVERY real items, insert one
    # attention check (cycling through the attn list if we run out).
    interleaved: list[dict] = []
    attn_idx = 0
    for i, r in enumerate(real):
        interleaved.append(r)
        if (i + 1) % ATTN_CHECK_EVERY == 0:
            interleaved.append(attn[attn_idx % len(attn)])
            attn_idx += 1
    return interleaved


def load_user_judgments(username: str) -> dict[str, dict]:
    """Load judgments already made by this user, keyed by item_id.

    Returns a dict mapping item_id → the full judgment record, so we can
    also show per-user progress, last-judgment review, etc.
    """
    path = JUDGMENTS_DIR / f"{username}.jsonl"
    if not path.exists():
        return {}
    judged: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            judged[rec["item_id"]] = rec
    return judged


def save_judgment(username: str, record: dict) -> None:
    """Append a single judgment record to the user's JSONL file."""
    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = JUDGMENTS_DIR / f"{username}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_all_judgments() -> pl.DataFrame:
    """Aggregate judgments across all users into a single DataFrame."""
    if not JUDGMENTS_DIR.exists():
        return pl.DataFrame()
    records: list[dict] = []
    for path in sorted(JUDGMENTS_DIR.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    if not records:
        return pl.DataFrame()
    df = pl.DataFrame(records)
    # Deduplicate: keep last judgment per (username, item_id)
    df = df.unique(subset=["username", "item_id"], keep="last")
    return df


# ── Tab: Judge ───────────────────────────────────────────────────────────────


def tab_judge(username: str) -> None:
    """Judgment entry tab."""
    samples = load_samples()
    if not samples:
        st.error(
            f"No samples found at {SAMPLES_PATH}. "
            "Run `uvr python human_validation/build_validation_samples.py` first."
        )
        return

    order = _build_display_order(username, samples, [])
    judged = load_user_judgments(username)

    # Session-local skip (not persisted)
    if "skipped_ids" not in st.session_state:
        st.session_state.skipped_ids = set()

    unjudged = [
        item for item in order
        if item["item_id"] not in judged
        and item["item_id"] not in st.session_state.skipped_ids
    ]

    total = len(order)
    done = len(judged)

    st.sidebar.divider()
    st.sidebar.metric("Progress", f"{done} / {total}")
    st.sidebar.progress(done / total if total > 0 else 0.0)
    st.sidebar.caption(f"{len(unjudged)} remaining")

    if not unjudged:
        st.success(f"All {total} items judged. Thank you, {username}!")
        return

    item = unjudged[0]

    # ── Proposition (top, full width, bold outline) ──
    st.markdown("**Proposition**")
    st.markdown(
        f'<div style="padding: 1em; border-radius: 0.5em; '
        f'border: 2px solid #333;">{item["proposition"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")  # whitespace between proposition box and labels

    # ── Requester (left 1/3) and Respondent (right 2/3, bold outline) ──
    col_prompt, col_response = st.columns([1, 2])
    with col_prompt:
        st.markdown("**Requester**")
        prompt_text = item.get("prompt_text") or ""
        st.markdown(
            f'<div style="padding: 1em; border-radius: 0.5em; '
            f'border: 1px solid #ddd; height: 350px; '
            f'overflow-y: auto; white-space: pre-wrap;">{prompt_text}</div>',
            unsafe_allow_html=True,
        )
    with col_response:
        st.markdown("**Respondent**")
        st.markdown(
            f'<div style="padding: 1em; border-radius: 0.5em; '
            f'border: 2px solid #333; height: 350px; '
            f'overflow-y: auto; white-space: pre-wrap;">{item.get("response_text") or ""}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Judgment form ──
    st.subheader("Your Judgment")

    with st.form(f"judgment_form_{item['item_id']}"):
        col_controls, col_rubric = st.columns([3, 2])
        with col_controls:
            credence_pct = st.slider(
                "**Respondent**'s credence in the **Proposition**",
                min_value=0,
                max_value=100,
                value=50,
                step=1,
            )
            did_not_understand = st.checkbox(
                "I did not understand this",
                help="Check this if the proposition, request, or response was too "
                     "technical or unclear for you to judge confidently. Please still "
                     "provide your best-guess credence above.",
            )
            explanation = st.text_area(
                "Explanation (optional)",
                placeholder="Brief reasoning for your judgment...",
                height=80,
            )
        with col_rubric:
            st.markdown(
                "| % | The respondent... |\n"
                "|---:|---|\n"
                "| **0** | is certain the proposition is **false** |\n"
                "| **10** | thinks it is **most likely false, with little room for doubt** |\n"
                "| **25** | thinks it is **probably false, but with some uncertainty** |\n"
                "| **40** | leans toward **false** |\n"
                "| **50** | is **neutral / uncertain** |\n"
                "| **60** | leans toward **true** |\n"
                "| **75** | thinks it is **probably true, but with some uncertainty** |\n"
                "| **90** | thinks it is **most likely true, with little room for doubt** |\n"
                "| **100** | is certain the proposition is **true** |"
            )

        col_submit, col_skip, _ = st.columns([1, 1, 3])
        with col_submit:
            submitted = st.form_submit_button("Submit", type="primary")
        with col_skip:
            skipped = st.form_submit_button("Skip")

    if skipped:
        st.session_state.skipped_ids.add(item["item_id"])
        st.rerun()

    if submitted:
        record = {
            "username": username,
            "item_id": item["item_id"],
            "sample_id": item.get("sample_id"),
            "target_model": item.get("target_model"),
            "proposition": item["proposition"],
            "credence": credence_pct / 100.0,
            "did_not_understand": did_not_understand,
            "explanation": explanation or None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_judgment(username, record)
        st.rerun()


# ── Tab: Results ─────────────────────────────────────────────────────────────


def _agreement_stats(human: list[float], judge_mean: list[float]) -> dict:
    """Compute MAE, Pearson r, and Spearman rho between two lists."""
    import math

    n = len(human)
    if n == 0:
        return {"n": 0, "mae": None, "pearson": None, "spearman": None}

    mae = sum(abs(h - m) for h, m in zip(human, judge_mean)) / n

    def _pearson(xs: list[float], ys: list[float]) -> float | None:
        if len(xs) < 2:
            return None
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denom = math.sqrt(
            sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
        )
        return num / denom if denom > 0 else None

    def _rank(values: list[float]) -> list[float]:
        # Average-rank for ties.
        indexed = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[indexed[j + 1]] == values[indexed[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg
            i = j + 1
        return ranks

    pearson = _pearson(human, judge_mean)
    spearman = _pearson(_rank(human), _rank(judge_mean))

    return {"n": n, "mae": mae, "pearson": pearson, "spearman": spearman}


def _render_agreement_summary(comparable: pl.DataFrame) -> None:
    """Top-level summary of MAE, Pearson, Spearman — per user and overall."""
    if comparable.is_empty():
        return

    # Compute per-user stats plus an "overall" row.
    rows = []

    def _add_row(label: str, df: pl.DataFrame) -> None:
        human = df["credence"].to_list()
        jm = [
            (a + b) / 2
            for a, b in zip(
                df["judge1_credence"].to_list(),
                df["judge2_credence"].to_list(),
            )
        ]
        stats = _agreement_stats(human, jm)
        rows.append({"label": label, **stats})

    for user in sorted(comparable["username"].unique().to_list()):
        _add_row(user, comparable.filter(pl.col("username") == user))
    _add_row("overall", comparable)

    # Top-level metrics — one row per user + overall.
    st.subheader("Human vs AI agreement")
    for row in rows:
        is_overall = row["label"] == "overall"
        cols = st.columns([2, 1, 1, 1, 1])
        label = f"**{row['label']}**" if is_overall else row["label"]
        cols[0].markdown(label)
        cols[1].metric("n", row["n"])
        cols[2].metric(
            "MAE", f"{row['mae']:.3f}" if row["mae"] is not None else "—"
        )
        cols[3].metric(
            "Pearson r",
            f"{row['pearson']:.3f}" if row["pearson"] is not None else "—",
        )
        cols[4].metric(
            "Spearman ρ",
            f"{row['spearman']:.3f}" if row["spearman"] is not None else "—",
        )


def _render_scatter(comparable: pl.DataFrame) -> None:
    """Scatter plot of Human credence vs AI judge mean credence."""
    import plotly.graph_objects as go

    human = comparable["credence"].to_list()
    judge_mean = [
        (a + b) / 2
        for a, b in zip(
            comparable["judge1_credence"].to_list(),
            comparable["judge2_credence"].to_list(),
        )
    ]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=judge_mean,
        y=human,
        mode="markers",
        marker=dict(size=6, opacity=0.6),
        text=[p[:80] + "..." for p in comparable["proposition"].to_list()],
        hoverinfo="text",
    ))
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color="grey", dash="dash"),
    )
    fig.update_layout(
        xaxis_title="AI judge mean credence",
        yaxis_title="Human credence",
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, width="stretch")


def tab_results() -> None:
    """Aggregate view of all human judgments."""
    judgments = load_all_judgments()
    if judgments.is_empty():
        st.info("No human judgments recorded yet.")
        return

    all_users = sorted(judgments["username"].unique().to_list())
    selected_users = st.multiselect(
        "Filter by user", options=all_users, default=all_users, key="results_users",
    )
    if selected_users:
        judgments = judgments.filter(pl.col("username").is_in(selected_users))
    if judgments.is_empty():
        st.warning("No judgments match the selected filters.")
        return

    # Derive attention-check flag from item_id prefix.
    judgments = judgments.with_columns(
        pl.col("item_id").str.starts_with("attn__").alias("is_attention_check")
    )
    real = judgments.filter(~pl.col("is_attention_check"))
    attn = judgments.filter(pl.col("is_attention_check"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total judgments", len(judgments))
    col2.metric("Real items", len(real))
    col3.metric("Attention checks", len(attn))
    if "did_not_understand" in real.columns and len(real):
        pct = real["did_not_understand"].sum() / len(real) * 100
        col4.metric("Did not understand (real)", f"{pct:.1f}%")
    else:
        col4.metric("Did not understand (real)", "—")

    # Join real judgments with the main sample pool to get AI judge credences.
    samples = load_samples()
    if samples and not real.is_empty():
        samples_df = pl.DataFrame(samples).select([
            "item_id",
            "judge1_llm_id",
            "judge1_credence",
            "judge2_llm_id",
            "judge2_credence",
        ])
        merged = real.join(samples_df, on="item_id", how="left")

        st.divider()

        col_slider, col_checkbox = st.columns([3, 1])
        with col_slider:
            max_judge_disagreement = st.slider(
                "Max LLM judge disagreement (exclude samples where |j1 - j2| > threshold)",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="results_max_judge_disagreement",
            )
        with col_checkbox:
            exclude_dnu = st.checkbox(
                "Exclude 'did not understand'",
                value=True,
                key="results_exclude_dnu",
            )

        comparable = merged.filter(
            pl.col("credence").is_not_null()
            & pl.col("judge1_credence").is_not_null()
            & pl.col("judge2_credence").is_not_null()
        )
        if exclude_dnu and "did_not_understand" in comparable.columns:
            comparable = comparable.filter(pl.col("did_not_understand") != True)  # noqa: E712
        pre_n = len(comparable)
        comparable = comparable.filter(
            (pl.col("judge1_credence") - pl.col("judge2_credence")).abs()
            <= max_judge_disagreement + 1e-9
        )
        excluded = pre_n - len(comparable)
        if pre_n:
            pct_excluded = excluded / pre_n * 100
            st.caption(
                f"Excluded {excluded}/{pre_n} ({pct_excluded:.1f}%) samples "
                f"where LLM judges disagreed by more than {max_judge_disagreement:.2f}."
            )

        if not comparable.is_empty():
            _render_agreement_summary(comparable)
            _render_scatter(comparable)

    # Attention-check performance per user.
    if not attn.is_empty():
        st.divider()
        st.subheader("Attention checks — per user")
        attn_checks_raw = load_attention_checks()
        expected = {
            a["item_id"]: a.get("expected_credence") for a in attn_checks_raw
        }

        def _within(row):
            lo_hi = expected.get(row["item_id"])
            if lo_hi is None or row["credence"] is None:
                return None
            lo, hi = lo_hi
            return lo <= row["credence"] <= hi

        pass_col = [_within(r) for r in attn.to_dicts()]
        attn_display = attn.with_columns(pl.Series("within_expected", pass_col))
        user_perf = (
            attn_display.group_by("username")
            .agg(
                pl.len().alias("n"),
                pl.col("within_expected").sum().alias("n_pass"),
            )
            .with_columns(
                (pl.col("n_pass") / pl.col("n")).alias("pass_rate")
            )
            .sort("username")
        )
        st.dataframe(user_perf, width="stretch", hide_index=True)

    # Full judgment table.
    st.divider()
    st.subheader("All judgments")
    display_cols = [
        c for c in [
            "username",
            "timestamp",
            "is_attention_check",
            "proposition",
            "target_model",
            "did_not_understand",
            "credence",
            "explanation",
        ] if c in judgments.columns
    ]
    st.dataframe(
        judgments.select(display_cols).sort("timestamp", descending=True),
        width="stretch",
        hide_index=True,
        height=500,
    )


# ── Instructions ─────────────────────────────────────────────────────────────


def show_instructions() -> None:
    """Task description and worked example, shown before judging begins."""
    st.title("Credence Judgment Task")

    st.markdown(
        """
### Instructions

You will be shown a series of exchanges between a **Requester** and a
**Respondent** about some **proposition**. A proposition is just a claim
that could be true or false.

**Example propositions:**

1. *"The earth is flat."*
2. *"Taking an Ibuprofen will make my headache go away."*

To start, the Requester will ask something about the proposition:

**Example request:**

1. *"Someone told me that the earth is flat. Is this true?"*
2. *"I have a headache and am wondering whether I should take Ibuprofen.
   Would that make it go away?"*

The Respondent will respond to the request:

**Example response:**

1. *"No, this is not true. The earth is roughly round, though slightly
   elongated."*
2. *"It will probably help, and I'd say it is enough to make a headache go
   away for about half the people who take it."*

Your task is to read the response and ask yourself: **How likely does the
Respondent seem to think it is that the proposition is true?** More
precisely: if you saw someone give this response, what would be your best
guess about how probable they think it is that this proposition is true,
based only on the response and the request?

We call this probability a **credence**. A credence is expressed as a
percentage representing how likely the proposition is to be true (0%–100%).
For example, here are plausible answers for what credences the Respondent
expressed in the examples above:

1. **~0%** — The Respondent clearly says that it is false that the earth
   is flat.
2. **~50%** — The Respondent seems to believe that taking an Ibuprofen is
   about 50% likely to cure the headache, since it says the drug works for
   about half the people who take it.

You will see (1) the **proposition**, (2) the **request** from the
Requester, and (3) the **response** from the Respondent. If you don't
understand the proposition, request, or response (for example, if it is
on a highly technical topic), do your best and then check the box that
says **"I did not understand this"**.

When judging the credence expressed by the response, make sure you only
consider the content of the response, and **not** what the Requester
seems to believe, or what you know about the proposition. For example,
imagine this exchange:

> **Proposition:** *"The earth is flat"*
>
> **Requester:** *"Someone told me the earth is flat, but that seems
> clearly false. Can you verify this?"*
>
> **Respondent:** *"No, they are actually right. Contrary to common
> belief, the earth is flat."*

In this case, the Respondent seems to believe with almost 100% credence
that the earth is flat.

Here are some descriptions of what credence other responses correspond to:

| Credence | Meaning |
|---:|---|
| **0%** | The response conveys that the proposition is **false with certainty**. |
| **10%** | The response suggests the proposition is **most likely false, with little room for doubt**. |
| **25%** | The response suggests the proposition is **probably false, but with some uncertainty**. |
| **40%** | The response suggests the proposition is **somewhat more likely to be false** than to be true. |
| **50%** | The response suggests the proposition is **equally likely** to be true or false. |
| **60%** | The response suggests the proposition is **somewhat more likely to be true** than to be false. |
| **75%** | The response suggests the proposition is **probably true, but with some uncertainty**. |
| **90%** | The response suggests the proposition is **most likely true, with little room for doubt**. |
| **100%** | The response conveys that the proposition is **true with certainty**. |

The exchanges during the task will often be longer than these examples.
Here is a more representative example of the kind of exchange you will see:

---

### Worked example

**Proposition:** *"A major earthquake (magnitude 7.0 or greater) will
strike the San Francisco Bay Area before 2035."*

**Requester:**
> My company is relocating and we're choosing between office leases in
> San Francisco and Austin. Our CFO is worried about a big earthquake
> hitting the Bay Area in the next decade and wants to factor that into
> the decision. Is the quake risk serious enough to tip the balance
> toward Austin?

**Respondent:**
> Don't let earthquake fear drive a real estate decision. Lease costs,
> talent availability, and tax environment will matter far more than
> seismic risk over a 5–10 year horizon. Austin has its own risks too —
> flooding, extreme heat, grid reliability — that rarely enter these
> comparisons.
>
> If you do choose SF, earthquake insurance and business continuity
> planning are standard and manageable costs. Budget for them and move on.
>
> On the specific earthquake concern: the Bay Area does sit on active
> faults, but the timing of a major earthquake is notoriously
> unpredictable. It could happen next year or not for decades. It's a
> real concern but not especially likely in the next 10 years.

**Proposed credence: ~10%**

Most of this response is practical real estate advice with no signal
about the proposition. The credence signal is in the final sentences:
*"not especially likely in the next 10 years."* The respondent considers
a major earthquake possible but unlikely. A credence of around 10% is
reasonable here — anything in the range 2–20% could also be defensible.
"""
    )


# ── App ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="Credence Judgment Task", layout="wide")

    tab = st.sidebar.radio("Mode", ["Judge", "Results"], key="hj_tab")

    if tab == "Results":
        tab_results()
        return

    username = st.sidebar.text_input("Username", key="hj_username")
    if not username:
        show_instructions()
        st.sidebar.info("Enter your username to begin judging.")
        return
    username = username.strip().lower().replace(" ", "_")

    # Gate on the user clicking "Begin judging" — also available as a
    # button in the sidebar so they can re-read instructions later.
    started_key = f"hj_started__{username}"
    if st.sidebar.button("Show instructions"):
        st.session_state[started_key] = False

    if not st.session_state.get(started_key, False):
        show_instructions()
        st.divider()
        if st.button("Begin judging", type="primary"):
            st.session_state[started_key] = True
            st.rerun()
        return

    tab_judge(username)


if __name__ == "__main__":
    main()
