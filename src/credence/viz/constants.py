"""Visualization constants for Round 2 credence experiments."""

# Consensus threshold for prompt attribute averaging (viz-only analysis constant)
AGREEMENT_THRESHOLD = 0.2

# --- Calibration bucket definitions (from preregistration) ---

BUCKET_ORDER = [
    "unequivocally_false",
    "likely_false",
    "uncertain",
    "likely_true",
    "unequivocally_true",
]

BUCKET_LABELS = {
    "unequivocally_false": "Unequiv. False",
    "likely_false": "Likely False",
    "uncertain": "Uncertain",
    "likely_true": "Likely True",
    "unequivocally_true": "Unequiv. True",
}

# (lo, hi) range values
BUCKET_RANGES: dict[str, tuple[float, float]] = {
    "unequivocally_false": (0.0, 0.05),
    "likely_false": (0.05, 0.45),
    "uncertain": (0.45, 0.55),
    "likely_true": (0.55, 0.95),
    "unequivocally_true": (0.95, 1.0),
}

# --- Pass thresholds ---

JUDGE_AGREEMENT_TARGET = 0.85
TEST_RETEST_SPEARMAN_TARGET = 0.7
TEST_RETEST_MAD_TARGET = 0.05

# --- Colors ---

PROVIDER_COLORS: dict[str, str] = {
    "anthropic": "#D97757",
    "openai": "#10A37F",
    "google": "#4285F4",
    "deepseek": "#4A90D9",
    "moonshot": "#E6553A",
}

DOMAIN_COLORS: dict[str, str] = {
    "contested_social_science": "#e377c2",
    "frontier_natural_science": "#2ca02c",
    "prediction_market": "#ff7f0e",
    "ai_claims": "#9467bd",
    "historical_facts": "#8c564b",
    "moral_claims": "#d62728",
    "nutrition_health": "#17becf",
    "paranormal_claims": "#7f7f7f",
    "philosophical_propositions": "#bcbd22",
    "politically_polarizing": "#1f77b4",
    "genuinely_uncertain": "#636EFA",
    "calibration_anchors": "#FF6692",
    "china_west_contentious": "#B6E880",
}

MODEL_GROUP_COLORS: dict[str, str] = {
    "western": "#636EFA",
    "chinese": "#EF553B",
}

CHINESE_PROVIDERS = {"deepseek", "moonshot"}

CATEGORY_COLORS: dict[str, str] = {
    "unequivocally_true": "#2ecc71",
    "unequivocally_false": "#e74c3c",
    "likely_true": "#3498db",
    "likely_false": "#e67e22",
    "uncertain": "#9b59b6",
}

# --- Bootstrap ---

BOOTSTRAP_N = 1000
BOOTSTRAP_CI_LEVEL = 95
BOOTSTRAP_SEED = 42
