from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    bbox_xyxy: tuple[float, float, float, float]
    track_id: int | None = None
    source: str = "main"
    owner_person_id: int | None = None
    hardhat_state: str | None = None  # "with_hard_hat" | "without_hard_hat" | None


@dataclass
class ViolationEvent:
    event_id: int
    frame_idx: int
    timestamp_sec: float
    person_track_id: int
    event_type: str
    no_hardhat_streak: int
    no_hardhat_duration_sec: float
    person_bbox: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

