"""Optional geometric confirmation: person box must have head/hardhat in upper body region."""

from __future__ import annotations

from .types import Detection


def person_confirmed_by_head_or_hardhat(
    person_xyxy: tuple[float, float, float, float],
    headlike_detections: list[Detection],
) -> bool:
    """True if some head/hardhat has bbox center inside person box and in upper 50% vertical band."""
    px1, py1, px2, py2 = person_xyxy
    ph = max(1e-6, py2 - py1)
    y_mid_upper = py1 + 0.5 * ph

    for det in headlike_detections:
        if det.cls_name not in ("head", "hardhat"):
            continue
        x1, y1, x2, y2 = det.bbox_xyxy
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        if not (px1 <= cx <= px2 and py1 <= cy <= py2):
            continue
        if cy > y_mid_upper + 1e-6:
            continue
        return True
    return False
