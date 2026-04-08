"""Data loading for the visualization app.

All loaders use @st.cache_data and return empty DataFrames when files are missing,
so tabs can degrade gracefully.
"""

from pathlib import Path

import polars as pl
import streamlit as st

from credence.core.export import RESULTS_DIR

_VALIDATION_EXPERIMENTS = {"calibration", "negation", "monotonicity", "china"}


@st.cache_data
def _read_parquet(path: str) -> pl.DataFrame:
    """Read a parquet file. Returns empty DataFrame if missing."""
    p = Path(path)
    if not p.exists():
        return pl.DataFrame()
    return pl.read_parquet(p)


def _load_results(experiment: str) -> pl.DataFrame:
    """Load results parquet for a given experiment."""
    if experiment in _VALIDATION_EXPERIMENTS:
        path = RESULTS_DIR / "validation" / f"{experiment}.parquet"
        return _read_parquet(str(path))
    if experiment == "exploration":
        return _load_exploration_parquets()
    path = RESULTS_DIR / f"{experiment}.parquet"
    return _read_parquet(str(path))


@st.cache_data
def _load_exploration_parquets() -> pl.DataFrame:
    """Load and concatenate all per-domain exploration parquet files."""
    files = sorted((RESULTS_DIR / "exploration").glob("*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal")


def _add_credence_consensus(df: pl.DataFrame) -> pl.DataFrame:
    """Compute consensus_credence from judge1/judge2 credences.

    Consensus = mean of judge credences when both are informative and
    differ by at most AGREEMENT_THRESHOLD. Otherwise null.
    """
    from credence.viz.constants import AGREEMENT_THRESHOLD

    cols = df.columns
    has_judges = "judge1_credence" in cols and "judge2_credence" in cols
    has_informative = "judge1_informative" in cols and "judge2_informative" in cols

    if not has_judges:
        return df

    both_informative = pl.lit(True)
    if has_informative:
        both_informative = pl.col("judge1_informative") & pl.col("judge2_informative")

    both_not_null = pl.col("judge1_credence").is_not_null() & pl.col("judge2_credence").is_not_null()
    within_threshold = (
        (pl.col("judge1_credence") - pl.col("judge2_credence")).abs()
        <= AGREEMENT_THRESHOLD + 1e-9
    )

    return df.with_columns(
        pl.when(both_informative & both_not_null & within_threshold)
        .then((pl.col("judge1_credence") + pl.col("judge2_credence")) / 2)
        .otherwise(None)
        .alias("consensus_credence")
    )


@st.cache_data
def load_calibration(run_id: int | None = None) -> pl.DataFrame:
    """Load calibration results, optionally filtered to a single run."""
    df = _add_credence_consensus(_load_results("calibration"))
    if df.is_empty() or run_id is None:
        return df
    if "run_id" in df.columns:
        return df.filter(pl.col("run_id") == run_id)
    return df


def _add_sided_consensus(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Compute consensus for a sided judge pair (e.g. p_judge1_credence -> p_consensus)."""
    from credence.viz.constants import AGREEMENT_THRESHOLD

    j1 = f"{prefix}_judge1_credence"
    j2 = f"{prefix}_judge2_credence"
    j1_inf = f"{prefix}_judge1_informative"
    j2_inf = f"{prefix}_judge2_informative"

    if j1 not in df.columns or j2 not in df.columns:
        return df

    both_informative = pl.lit(True)
    if j1_inf in df.columns and j2_inf in df.columns:
        both_informative = pl.col(j1_inf) & pl.col(j2_inf)

    both_not_null = pl.col(j1).is_not_null() & pl.col(j2).is_not_null()
    within_threshold = (pl.col(j1) - pl.col(j2)).abs() <= AGREEMENT_THRESHOLD + 1e-9

    return df.with_columns(
        pl.when(both_informative & both_not_null & within_threshold)
        .then((pl.col(j1) + pl.col(j2)) / 2)
        .otherwise(None)
        .alias(f"{prefix}_consensus")
    )


@st.cache_data
def load_negation() -> pl.DataFrame:
    """Load negation experiment data."""
    df = _add_credence_consensus(_load_results("negation"))
    df = _add_sided_consensus(df, "p")
    df = _add_sided_consensus(df, "notp")
    return df


def _add_monotonicity_credences(df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-proposition consensus credences for monotonicity data.

    Discovers prop{i}_judge{j}_credence columns, computes consensus per position
    (same threshold logic as other experiments), and builds a 'credences' list column.
    """
    from credence.viz.constants import AGREEMENT_THRESHOLD

    max_props = 0
    while f"prop{max_props + 1}_judge1_credence" in df.columns:
        max_props += 1

    if max_props == 0:
        return df

    def _row_credences(row: dict) -> list[float | None]:
        result: list[float | None] = []
        for i in range(1, max_props + 1):
            j1 = row.get(f"prop{i}_judge1_credence")
            j2 = row.get(f"prop{i}_judge2_credence")
            j1_inf = row.get(f"prop{i}_judge1_informative", True)
            j2_inf = row.get(f"prop{i}_judge2_informative", True)
            if (
                j1 is not None
                and j2 is not None
                and j1_inf
                and j2_inf
                and abs(j1 - j2) <= AGREEMENT_THRESHOLD + 1e-9
            ):
                result.append((j1 + j2) / 2)
            else:
                result.append(None)
        return result

    credences = [_row_credences(row) for row in df.iter_rows(named=True)]
    return df.with_columns(pl.Series("credences", credences))


@st.cache_data
def load_monotonicity() -> pl.DataFrame:
    """Load monotonicity experiment data."""
    df = _load_results("monotonicity")
    if df.is_empty():
        return df
    return _add_monotonicity_credences(df)


@st.cache_data
def load_china() -> pl.DataFrame:
    """Load China comparison experiment data."""
    return _add_credence_consensus(_load_results("china"))


@st.cache_data
def load_exploration() -> pl.DataFrame:
    """Load exploration results with computed consensus columns."""
    df = _load_results("exploration")
    if df.is_empty():
        return df
    df = _add_credence_consensus(df)
    return _add_prompt_attribute_consensus(df)


def _add_prompt_attribute_consensus(df: pl.DataFrame) -> pl.DataFrame:
    """Add consensus columns for prompt attributes (average if judges agree within threshold).

    Dynamically discovers prompt_judgeN_* and evidence_judgeN_* columns and computes
    pairwise consensus across all judge pairs. Works with any number of judges.
    """
    from credence.core.schemas import NEW_EVIDENCE_ATTRIBUTE_NAMES, VALENCE_ATTRIBUTE_NAMES
    from credence.viz.constants import AGREEMENT_THRESHOLD

    # (attribute_name, column_prefix) pairs for all prompt attribute scorers
    attr_prefixes: list[tuple[str, str]] = [
        (attr, "prompt_judge") for attr in VALENCE_ATTRIBUTE_NAMES
    ] + [
        (attr, "evidence_judge") for attr in NEW_EVIDENCE_ATTRIBUTE_NAMES
        if attr == "new_evidence_score"  # only numeric attrs get consensus
    ]

    for attr, prefix in attr_prefixes:
        # Discover all judge columns for this attribute
        judge_cols = sorted(
            col for col in df.columns
            if col.startswith(prefix) and col.endswith(f"_{attr}")
        )

        if len(judge_cols) < 2:
            if len(judge_cols) == 1:
                df = df.with_columns(pl.col(judge_cols[0]).alias(f"consensus_{attr}"))
            continue

        # Mean of all judge values where all pairwise differences are within threshold
        not_null = pl.lit(True)
        for col in judge_cols:
            not_null = not_null & pl.col(col).is_not_null()

        within_threshold = pl.lit(True)
        for i in range(len(judge_cols)):
            for j in range(i + 1, len(judge_cols)):
                within_threshold = within_threshold & (
                    (pl.col(judge_cols[i]) - pl.col(judge_cols[j])).abs()
                    <= AGREEMENT_THRESHOLD + 1e-9
                )

        mean_expr = pl.lit(0.0)
        for col in judge_cols:
            mean_expr = mean_expr + pl.col(col)
        mean_expr = mean_expr / len(judge_cols)

        df = df.with_columns(
            pl.when(not_null & within_threshold)
            .then(mean_expr)
            .otherwise(None)
            .alias(f"consensus_{attr}")
        )
    return df


@st.cache_data
def load_all_for_agreement() -> pl.DataFrame:
    """Load calibration data for judge agreement analysis.

    Filters to samples where both judges marked the response as informative.
    """
    df = load_calibration()
    if df.is_empty():
        return df

    # Filter to samples where all judges marked the response as informative
    informative_cols = sorted(
        col for col in df.columns
        if col.startswith("judge") and col.endswith("_informative")
    )
    if informative_cols:
        mask = pl.lit(True)
        for col in informative_cols:
            mask = mask & pl.col(col)
        df = df.filter(mask)
    return df
