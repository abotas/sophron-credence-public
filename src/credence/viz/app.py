"""Streamlit visualization for Round 2 credence elicitation experiments.

Usage:
    uvr streamlit run src/credence/viz/app.py
"""

from pathlib import Path

import streamlit as st



# --- Navigation helpers ---

def _nav_link(label, section, tab):
    """Generate a markdown link that navigates to a section/tab."""
    return f"[{label}](?section={section}&tab={tab})"


# --- Section renderers ---

def _render_overview():
    """Render the overview/landing page."""
    st.title("Measuring AI Belief")

    st.markdown("""
**What beliefs do AI models express?** We built an automated pipeline to extract
probabilistic beliefs about arbitrary propositions from natural language model outputs. The purpose of this is to supply a scalable and automatic way to assess
and make transparent what beliefs models express across a range of topics, and to test
how that depends on factors like the framing of the user-prompt and the nature of the
topic.

On this provisional page, we demonstrate our initial results. We surveyed frontier
models on ~1,250 propositions. These results can be seen here in two main parts:

**Part 1:** Results validating our method with a series of robustness checks
specified in the tabs.

**Part 2:** Substantive results showing expressed model credences across topics, how
sensitive they are to framing effects (e.g. perceived user-belief about the topic),
and differences between models.

This page presents results from **Round 2** of our validation experiments. Round 1
(January 2026) established that the pipeline produces stable, calibrated, and
discriminating credence estimates. Round 2 introduces finer-grained calibration targets,
new coherence tests (negation consistency and monotonicity), and re-validates
reliability and validity under an updated pipeline configuration.
    """)

    st.subheader("Method")
    st.markdown("""
1. **Proposition construction** — Claims designed to fall within a target credence range,
   manually curated to minimize ambiguity.
2. **Prompt generation** — Elicitor models generate 32 diverse naturalistic prompts per
   proposition (16 per elicitor), varying in framing, tone, and implied user belief.
3. **Target model response** — The model under test responds to each prompt.
4. **Multi-judge evaluation** — Two judge models independently score the expressed
   credence (0–1). If both judges are informative and agree within 0.2, we take
   the average as the consensus credence; otherwise the sample is excluded.
    """)

    st.divider()

    st.subheader("What's in this app")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Part 1: Validation**")
        st.caption(
            "Preregistered checks that the pipeline produces stable, calibrated, "
            "coherent, and discriminating credence estimates."
        )
        validation_links = f"""
| | Check |
|---|---|
| V1 | {_nav_link("Calibration", "validation", "calibration")} — Do credences match expected ranges across five buckets? |
| V2 | {_nav_link("Judge Coherence", "validation", "coherence")} — Do judges respect negation consistency and monotonicity? |
| V3 | {_nav_link("Test-Retest", "validation", "test_retest")} — Do repeated pipeline runs yield similar estimates? |
| V4 | {_nav_link("Judge Agreement", "validation", "judge_agreement")} — Do different judges produce similar scores? |
| V5 | {_nav_link("Known Divergence", "validation", "known_group")} — Do Chinese/Western models diverge on sensitive topics? |
| | {_nav_link("Inspect", "validation", "inspect")} — Deep-dive into individual propositions and samples |
"""
        st.markdown(validation_links)

    with col2:
        st.markdown("**Part 2: Exploration**")
        st.caption(
            "Substantive results showing expressed model credences across topics, "
            "sensitivity to prompt framing, and differences between models."
        )
        exploration_links = f"""
| | Tab |
|---|---|
| E1 | {_nav_link("What Models Believe", "exploration", "beliefs")} — Extremity and dispersion of model credences |
| E2 | {_nav_link("Prompt Sensitivity", "exploration", "sensitivity")} — How framing affects expressed credence |
| E3 | {_nav_link("Cross-Model Agreement", "exploration", "model_agreement")} — Pairwise model correlations |
| | {_nav_link("Inspect Credences", "exploration", "inspect")} — Deep-dive into individual propositions and samples |
| | {_nav_link("Inspect Prompt Attributes", "exploration", "inspect_attrs")} — Deep-dive into prompt attribute judgments |
"""
        st.markdown(exploration_links)

    st.divider()

    st.subheader("Study details")
    st.markdown("""
**Round 2 models:**

| Role | Experiments 1–5 | Experiment 6 (Known Divergence) |
|------|----------------|---------------------------|
| **Elicitors** | GPT-5-mini, Claude Sonnet 4.6 | GPT-5-mini, Claude Sonnet 4.6 |
| **Target models** | GPT-5-mini | GPT-5-mini, Claude Sonnet 4.6, DeepSeek-chat, Kimi-K2.5 |
| **Judges** | GPT-5-mini, Claude Sonnet 4.6 | GPT-5-mini, Claude Sonnet 4.6 |

32 prompts per proposition (16 per elicitor). 2 independent pipeline runs.

Our Round 2 preregistration is available in the repository. The Round 1 preregistration
and design spec are available
[here](https://drive.google.com/file/d/1Z_mQGgAeWqXK-sdQANgboEe4PIJh3yWG/view?usp=sharing)
and [here](https://drive.google.com/file/d/1yJLkFneyA0dqSksHP8ij0grsQpXplbTh/view?usp=sharing).
    """)


def _render_validation(default_tab):
    """Render the validation section with tabs."""
    st.header("Part 1: Method Validation")

    from credence.viz.tabs.calibration import render as render_calibration
    from credence.viz.tabs.coherence import render as render_coherence
    from credence.viz.tabs.inspect_tab import render as render_inspect
    from credence.viz.tabs.judge_agreement import render as render_judge_agreement
    from credence.viz.tabs.known_group import render as render_known_group
    from credence.viz.tabs.test_retest import render as render_test_retest

    tab_map = {
        "calibration": ("V1 Calibration", render_calibration),
        "coherence": ("V2 Judge Coherence", render_coherence),
        "test_retest": ("V3 Test-Retest", render_test_retest),
        "judge_agreement": ("V4 Judge Agreement", render_judge_agreement),
        "known_group": ("V5 Known Divergence", render_known_group),
        "inspect": ("Inspect", render_inspect),
    }
    tab_keys = list(tab_map.keys())
    tab_labels = [tab_map[k][0] for k in tab_keys]

    default_idx = tab_keys.index(default_tab) if default_tab in tab_keys else 0
    tabs = st.tabs(tab_labels)

    for i, key in enumerate(tab_keys):
        with tabs[i]:
            tab_map[key][1]()


def _render_exploration(default_tab):
    """Render the exploration section with tabs."""
    st.header("Part 2: Exploration")

    from credence.viz.tabs.beliefs import render as render_beliefs
    from credence.viz.tabs.explore_inspect import render as render_explore_inspect
    from credence.viz.tabs.explore_inspect_attrs import render as render_explore_inspect_attrs
    from credence.viz.tabs.model_agreement import render as render_model_agreement
    from credence.viz.tabs.sensitivity import render as render_sensitivity

    tab_map = {
        "beliefs": ("E1 What Models Believe", render_beliefs),
        "sensitivity": ("E2 Prompt Sensitivity", render_sensitivity),
        "model_agreement": ("E3 Cross-Model Agreement", render_model_agreement),
        "inspect": ("Inspect Credences", render_explore_inspect),
        "inspect_attrs": ("Inspect Prompt Attributes", render_explore_inspect_attrs),
    }
    tab_keys = list(tab_map.keys())
    tab_labels = [tab_map[k][0] for k in tab_keys]

    tabs = st.tabs(tab_labels)

    for i, key in enumerate(tab_keys):
        with tabs[i]:
            tab_map[key][1]()


# --- Main ---

def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(page_title="Credence Visualization", layout="wide")

    # Force text wrapping in st.code blocks (instead of horizontal scroll)
    st.markdown(
        "<style>code { white-space: pre-wrap !important; word-wrap: break-word !important; }</style>",
        unsafe_allow_html=True,
    )

    # Read query params for deep linking
    params = st.query_params
    section_param = params.get("section", "")
    tab_param = params.get("tab", "")

    if section_param == "validation":
        default_section = "Part 1: Validation"
    elif section_param == "exploration":
        default_section = "Part 2: Exploration"
    else:
        default_section = "Overview"

    # Sidebar
    with st.sidebar:
        sections = ["Overview", "Part 1: Validation", "Part 2: Exploration"]
        section = st.radio("Section", sections, index=sections.index(default_section))


    # Route to section
    if section == "Overview":
        _render_overview()
    elif section == "Part 1: Validation":
        _render_validation(tab_param)
    elif section == "Part 2: Exploration":
        _render_exploration(tab_param)


def _cli_entry():
    """Entry point for `credence-viz` CLI command."""
    import sys

    from streamlit.web.cli import main as st_main

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    st_main()


if __name__ == "__main__":
    main()
