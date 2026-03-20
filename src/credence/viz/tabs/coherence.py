"""V2 Judge Coherence — Negation Consistency + Monotonicity."""

import streamlit as st


def render() -> None:
    """Render the combined judge coherence tab."""
    st.subheader("V2 Judge Coherence: Do judges respect basic probabilistic constraints?")
    st.caption(
        "Two coherence tests probe whether judge credence scores obey basic probabilistic "
        "rules that do not depend on ground truth. Negation consistency checks that "
        "J(P) + J(¬P) ≈ 1. Monotonicity checks that logically ordered propositions "
        "yield monotonically ordered credences."
    )

    neg_tab, mono_tab = st.tabs(["Negation Consistency", "Monotonicity"])

    with neg_tab:
        from credence.viz.tabs.negation import render as render_negation
        render_negation()

    with mono_tab:
        from credence.viz.tabs.monotonicity import render as render_monotonicity
        render_monotonicity()
