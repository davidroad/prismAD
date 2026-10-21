"""Frozen feature and information-track definitions."""

from __future__ import annotations

from dataclasses import dataclass


SEED = 42

CURRENT_FEATURES = (
    "visit_age",
    "PHC_Sex",
    "PHC_Education",
    "PHC_Race",
    "PHC_Ethnicity",
    "global_z",
    "PHC_MEM_z",
    "PHC_EXF_z",
    "PHC_LAN_z",
    "PHC_VSP_z",
    "n_domains",
)

DEMOGRAPHIC_FEATURES = CURRENT_FEATURES[:5]

STRICT_HISTORY_FEATURES = (
    "strict_history_events",
    "strict_history_span_years",
    "strict_history_domains",
    "strict_history_time_since_last_years",
    "strict_history_common_pace_per_5y",
    "strict_history_memory_last",
    "strict_history_memory_time_since_last",
    "strict_history_memory_pace_per_5y",
    "strict_history_memory_events",
    "strict_history_executive_last",
    "strict_history_executive_time_since_last",
    "strict_history_executive_pace_per_5y",
    "strict_history_executive_events",
    "strict_history_language_last",
    "strict_history_language_time_since_last",
    "strict_history_language_pace_per_5y",
    "strict_history_language_events",
    "strict_history_visuospatial_last",
    "strict_history_visuospatial_time_since_last",
    "strict_history_visuospatial_pace_per_5y",
    "strict_history_visuospatial_events",
)

CATEGORICAL_FEATURES = frozenset({"PHC_Sex", "PHC_Race", "PHC_Ethnicity"})

LOG1P_FEATURES = frozenset(
    {
        "strict_history_events",
        "strict_history_span_years",
        "strict_history_domains",
        "strict_history_time_since_last_years",
        "strict_history_memory_time_since_last",
        "strict_history_memory_events",
        "strict_history_executive_time_since_last",
        "strict_history_executive_events",
        "strict_history_language_time_since_last",
        "strict_history_language_events",
        "strict_history_visuospatial_time_since_last",
        "strict_history_visuospatial_events",
    }
)


@dataclass(frozen=True)
class TrackSpec:
    name: str
    current_features: tuple[str, ...]
    history_summary_features: tuple[str, ...]
    use_prs: bool
    use_sequence: bool
    description: str


TRACKS = {
    "A": TrackSpec(
        "A",
        DEMOGRAPHIC_FEATURES,
        (),
        True,
        False,
        "Demographics plus PRS",
    ),
    "B": TrackSpec(
        "B",
        CURRENT_FEATURES,
        (),
        False,
        False,
        "Demographics plus current cognition, without PRS",
    ),
    "C": TrackSpec(
        "C",
        CURRENT_FEATURES,
        (),
        True,
        False,
        "Demographics plus current cognition and PRS",
    ),
    "D": TrackSpec(
        "D",
        CURRENT_FEATURES,
        STRICT_HISTORY_FEATURES,
        True,
        True,
        "Current phenotype, PRS and strict-past cognitive trajectory",
    ),
}

# Clinically motivated, prespecified GA2M interactions. Missing endpoints are
# filtered structurally when a track does not contain the required variables.
PAIR_ALLOWLIST = (
    ("visit_age", "global_z"),
    ("visit_age", "PHC_MEM_z"),
    ("PHC_Education", "global_z"),
    ("global_z", "PHC_MEM_z"),
    ("global_z", "PHC_EXF_z"),
    ("global_z", "strict_history_common_pace_per_5y"),
    ("global_z", "strict_history_memory_pace_per_5y"),
    ("global_z", "strict_history_executive_pace_per_5y"),
    ("global_z", "strict_history_events"),
    ("PHC_MEM_z", "strict_history_memory_last"),
    ("PHC_MEM_z", "strict_history_memory_pace_per_5y"),
    ("PHC_EXF_z", "strict_history_executive_last"),
    ("PHC_EXF_z", "strict_history_executive_pace_per_5y"),
    ("strict_history_span_years", "strict_history_common_pace_per_5y"),
)

EVENT_NUMERIC_FEATURES = (
    "value_z",
    "reported_se_z",
    "reported_se_missing",
    "precise",
    "precise_missing",
    "log1p_lag_z",
    "log1p_visit_num_z",
)

