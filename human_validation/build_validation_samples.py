"""Build a flat dataset of (proposition, prompt, response) samples for
human credence validation.

Loads all per-domain exploration parquets from results/round4/exploration/,
drops samples where either judge marked informative=False, and writes:

1. Full dataset → .data/human_judgments/samples_for_human_validation.jsonl.gz
2. Stratified 1k sample → human_validation/samples_for_human_validation_1k.jsonl.gz
   (100 per domain, ~17 per domain×model stratum)

Each row has a deterministic UUID5 item_id derived from (sample_id, target_model),
so any downstream tool (Qualtrics, Prolific, etc.) can join back to the
original pipeline data.
"""

import gzip
import json
import uuid
from pathlib import Path

import polars as pl

# Arbitrary fixed namespace for item_id generation. Do NOT change — if you do,
# all item_ids for existing annotations become orphaned.
_ITEM_ID_NS = uuid.UUID("c0ffee00-0000-4000-8000-000000000000")

# Seed for the stratified 1k sample. Documented here for reproducibility.
_SAMPLE_SEED = 42

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXPLORATION_DIR = _PROJECT_ROOT / "results" / "round4" / "exploration"
_FULL_OUTPUT_DIR = _PROJECT_ROOT / ".data" / "human_judgments"
_SAMPLE_OUTPUT_DIR = _PROJECT_ROOT / "human_validation"


def _item_id(sample_id: str, target_model: str) -> str:
    """Deterministic UUID5 derived from (sample_id, target_model)."""
    return str(uuid.uuid5(_ITEM_ID_NS, f"{sample_id}|{target_model}"))


def _derive_domain(df: pl.DataFrame) -> pl.DataFrame:
    """Add a 'domain' column derived from proposition_id in the source parquets.

    Since the minimal schema doesn't include domain, we re-derive it by
    joining back to the parquets on (sample_id, target_model).
    """
    parquet_files = sorted(_EXPLORATION_DIR.glob("*.parquet"))
    full = pl.concat([pl.read_parquet(f) for f in parquet_files], how="diagonal")
    domain_map = full.select(["sample_id", "target_model", "domain"]).unique()
    return df.join(domain_map, on=["sample_id", "target_model"], how="left")


def _stratified_sample(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """Stratified sample: equal allocation across (domain, target_model) strata.

    Within each stratum, samples are drawn randomly with the given seed.
    If a stratum has fewer rows than its allocation, all rows are taken
    and the shortfall is redistributed.
    """
    strata = df.group_by(["domain", "target_model"])
    n_strata = strata.len().height
    per_stratum = n // n_strata
    remainder = n % n_strata

    parts = []
    leftover_budget = 0
    for (domain, model), group in strata:
        alloc = per_stratum + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        if len(group) <= alloc:
            parts.append(group)
            leftover_budget += alloc - len(group)
        else:
            parts.append(group.sample(n=alloc, seed=seed, shuffle=True))

    sampled = pl.concat(parts)

    # Redistribute shortfall: sample extra rows from the unselected pool.
    if leftover_budget > 0:
        already = set(sampled["item_id"].to_list())
        pool = df.filter(~pl.col("item_id").is_in(list(already)))
        if len(pool) > 0:
            extra = pool.sample(
                n=min(leftover_budget, len(pool)),
                seed=seed + 1,
                shuffle=True,
            )
            sampled = pl.concat([sampled, extra])

    # Final shuffle for presentation order.
    return sampled.sample(fraction=1.0, seed=seed + 2, shuffle=True)


def _write_jsonl_gz(df: pl.DataFrame, path: Path) -> None:
    """Write a DataFrame to a gzipped JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in df.iter_rows(named=True):
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parquet_files = sorted(_EXPLORATION_DIR.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet files found in {_EXPLORATION_DIR}")

    frames = [pl.read_parquet(f) for f in parquet_files]
    df = pl.concat(frames, how="diagonal")
    print(f"Loaded {len(df):,} rows from {len(parquet_files)} parquets")

    # Drop samples where either judge marked the response uninformative.
    before = len(df)
    df = df.filter(
        pl.col("judge1_informative").eq(True)
        & pl.col("judge2_informative").eq(True)
    )
    print(f"Filtered to {len(df):,} rows (dropped {before - len(df):,} uninformative)")

    # Derive item_id (deterministic UUID5).
    item_ids = [
        _item_id(sid, tm)
        for sid, tm in zip(df["sample_id"].to_list(), df["target_model"].to_list())
    ]
    df = df.with_columns(pl.Series("item_id", item_ids))

    # Minimal schema for output.
    out = df.select([
        "item_id",
        "sample_id",
        "target_model",
        "proposition",
        "prompt_text",
        "response_text",
        "judge1_llm_id",
        "judge1_credence",
        "judge2_llm_id",
        "judge2_credence",
    ])

    # 1. Full dataset → .data/human_judgments/
    full_path = _FULL_OUTPUT_DIR / "samples_for_human_validation.jsonl.gz"
    _write_jsonl_gz(out, full_path)
    print(f"Wrote {len(out):,} rows → {full_path}")

    # 2. Stratified 1k sample → human_validation/
    #    Need domain for stratification — derive it by joining back.
    out_with_domain = _derive_domain(out)
    sampled = _stratified_sample(out_with_domain, n=1000, seed=_SAMPLE_SEED)
    # Drop the domain column (not in the output schema; derivable from source).
    sampled = sampled.drop("domain")
    sample_path = _SAMPLE_OUTPUT_DIR / "samples_for_human_validation_1k.jsonl.gz"
    _write_jsonl_gz(sampled, sample_path)
    print(f"Wrote {len(sampled):,} rows → {sample_path}")

    # Show stratification breakdown.
    sampled_with_domain = _derive_domain(sampled)
    breakdown = (
        sampled_with_domain
        .group_by(["domain", "target_model"])
        .len()
        .sort(["domain", "target_model"])
    )
    print(f"\nStratification ({len(sampled):,} total):")
    for domain in sorted(breakdown["domain"].unique().to_list()):
        subset = breakdown.filter(pl.col("domain") == domain)
        total = subset["len"].sum()
        models = ", ".join(
            f"{row['target_model'].split('/')[-1]}={row['len']}"
            for row in subset.to_dicts()
        )
        print(f"  {domain}: {total} ({models})")


if __name__ == "__main__":
    main()
