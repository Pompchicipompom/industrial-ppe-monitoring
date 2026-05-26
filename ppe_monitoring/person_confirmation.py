"""Person confirmation modes: off / hard (head-only) / soft (multi-signal)."""

from __future__ import annotations

from typing import Any

from .person_head_confirmation import person_confirmed_by_head_or_hardhat
from .types import Detection


def resolve_person_confirmation_mode(filters: dict[str, Any]) -> str:
    """Return one of off | hard | soft. Legacy require_head_or_hardhat_for_person=true maps to hard."""
    mode = str(filters.get("person_confirmation_mode", "off")).strip().lower()
    if mode not in ("off", "hard", "soft"):
        mode = "off"
    if mode == "off" and bool(filters.get("require_head_or_hardhat_for_person", False)):
        mode = "hard"
    return mode


def person_soft_confirmed(
    person_xyxy: tuple[float, float, float, float],
    headlikes: list[Detection],
    person_conf: float,
    frame_w: int,
    frame_h: int,
    infer_hit_count: int,
    *,
    high_conf_threshold: float,
    min_aspect_hw: float,
    min_area_ratio: float,
    min_infer_hits: int,
) -> bool:
    """
    Soft OR logic:
    A) head/hardhat center in person bbox upper half
    B) person confidence >= high_conf_threshold
    C) height/width >= min_aspect_hw and area ratio >= min_area_ratio
    D) infer_hit_count >= min_infer_hits
    """
    if person_confirmed_by_head_or_hardhat(person_xyxy, headlikes):
        return True
    if person_conf >= high_conf_threshold:
        return True
    x1, y1, x2, y2 = person_xyxy
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    area_ratio = (w * h) / float(max(1, frame_w) * max(1, frame_h))
    if h / w >= min_aspect_hw and area_ratio >= min_area_ratio:
        return True
    if infer_hit_count >= min_infer_hits:
        return True
    return False
