"""Shared column names for frame_metrics.csv and stage timing in runtime_profile.

Used by pipeline (writer) and optionally by tools (reader validation) to avoid drift.
"""

from __future__ import annotations

# Stage keys passed to RuntimeProfiler.record_frame(stage_timings_ms=...)
STAGE_TIMING_FIELDS: tuple[str, ...] = (
    "main_infer_ms",
    "roi_infer_ms",
    "tracking_ms",
    "event_logic_ms",
    "render_ms",
    "io_ms",
    "total_loop_ms",
)

_METRICS_PREFIX: tuple[str, ...] = (
    "frame_idx",
    "timestamp_sec",
    "motion_ratio",
    "sampled",
    "did_infer",
    "infer_ms",
    "loop_ms",
)

_METRICS_SUFFIX: tuple[str, ...] = (
    "active_persons",
    "active_violations",
    "events_this_frame",
    "events_total",
)


def metrics_csv_fieldnames() -> list[str]:
    """Full header row for frame_metrics.csv (must match pipeline row order)."""
    return list(_METRICS_PREFIX) + list(STAGE_TIMING_FIELDS) + list(_METRICS_SUFFIX)


def validate_metrics_csv_header(fieldnames: list[str] | tuple[str, ...] | None) -> None:
    expected = metrics_csv_fieldnames()
    if fieldnames is None:
        raise ValueError("frame_metrics.csv: missing header row")
    got = list(fieldnames)
    if got != expected:
        raise ValueError(
            "frame_metrics.csv: header mismatch.\n"
            f"  expected: {expected}\n"
            f"  got:      {got}"
        )
