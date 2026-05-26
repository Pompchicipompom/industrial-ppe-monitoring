"""EventConsolidator: post-process raw violation events into deduplicated episodes.

Designed for production-like emission where multiple track_ids per video are allowed,
but track-id churn / temporal hold should not produce duplicates of the same physical
violation episode.

The consolidator is decoupled from the pipeline:
- input: list of RawEvent (with optional person_bbox);
- output: subset of input that represents one row per distinct episode.

Matching rules between an incoming candidate event and an active episode:
- same event_type AND time_gap <= merge_gap_seconds AND
  (a) same person_track_id, OR
  (b) bbox IoU >= spatial_merge_iou_threshold, OR
  (c) bbox center distance / max(bbox_side) <= spatial_merge_center_distance_ratio.

Reopen guard: a new episode for the same track/region is suppressed within
reopen_after_seconds after the previous episode's last seen ts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConsolidatorParams:
    merge_gap_seconds: float = 3.0
    spatial_merge_iou_threshold: float = 0.30
    spatial_merge_center_distance_ratio: float = 0.40
    clear_seconds: float = 4.0
    reopen_after_seconds: float = 10.0
    cooldown_seconds_per_track: float = 8.0
    max_events_per_track_per_episode: int = 1
    allow_multiple_tracks_per_video: bool = True


@dataclass
class RawEvent:
    video_id: str
    event_id: str
    frame_idx: int
    timestamp_sec: float
    person_track_id: int | None
    event_type: str
    bbox: tuple[float, float, float, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _iou(a: tuple | None, b: tuple | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _center_dist_ratio(a: tuple | None, b: tuple | None) -> float:
    if a is None or b is None:
        return float("inf")
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    scale = max(a[2] - a[0], a[3] - a[1], 1.0)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / scale


class EventConsolidator:
    def __init__(self, params: ConsolidatorParams | None = None) -> None:
        self.params = params or ConsolidatorParams()

    def _episode_matches(self, episode: dict, ev: RawEvent, params: ConsolidatorParams) -> bool:
        if ev.event_type != episode["event_type"]:
            return False
        gap = ev.timestamp_sec - episode["last_ts"]
        if gap < 0 or gap > params.merge_gap_seconds:
            return False
        if ev.person_track_id is not None and ev.person_track_id in episode["track_ids"]:
            return True
        if ev.bbox is not None and episode["last_bbox"] is not None:
            if _iou(ev.bbox, episode["last_bbox"]) >= params.spatial_merge_iou_threshold:
                return True
            if _center_dist_ratio(ev.bbox, episode["last_bbox"]) <= params.spatial_merge_center_distance_ratio:
                return True
        return False

    def _reopen_blocked(self, episode: dict, ev: RawEvent, params: ConsolidatorParams) -> bool:
        gap = ev.timestamp_sec - episode["last_ts"]
        if gap <= params.merge_gap_seconds or gap >= params.reopen_after_seconds:
            return False
        if ev.event_type != episode["event_type"]:
            return False
        if ev.person_track_id is not None and ev.person_track_id in episode["track_ids"]:
            return True
        if ev.bbox is not None and episode["last_bbox"] is not None:
            if _iou(ev.bbox, episode["last_bbox"]) >= params.spatial_merge_iou_threshold:
                return True
            if _center_dist_ratio(ev.bbox, episode["last_bbox"]) <= params.spatial_merge_center_distance_ratio:
                return True
        return False

    def consolidate_video(self, raw: list[RawEvent]) -> tuple[list[RawEvent], dict[str, int]]:
        params = self.params
        raw_sorted = sorted(raw, key=lambda r: (r.frame_idx, r.event_id))
        episodes: list[dict] = []
        emitted: list[RawEvent] = []
        merged_dupes = 0
        reopen_blocks = 0

        for ev in raw_sorted:
            matched = None
            for ep in episodes:
                if self._episode_matches(ep, ev, params):
                    matched = ep
                    break

            if matched is not None:
                if ev.person_track_id is not None:
                    matched["track_ids"].add(ev.person_track_id)
                if ev.bbox is not None:
                    matched["last_bbox"] = ev.bbox
                matched["last_frame"] = ev.frame_idx
                matched["last_ts"] = ev.timestamp_sec
                matched["count"] += 1
                merged_dupes += 1
                continue

            blocked = any(self._reopen_blocked(ep, ev, params) for ep in episodes)
            if blocked:
                reopen_blocks += 1
                continue

            episodes.append(
                {
                    "event_type": ev.event_type,
                    "track_ids": {ev.person_track_id} if ev.person_track_id is not None else set(),
                    "last_bbox": ev.bbox,
                    "last_frame": ev.frame_idx,
                    "last_ts": ev.timestamp_sec,
                    "opened_ts": ev.timestamp_sec,
                    "count": 1,
                }
            )
            emitted.append(ev)

        return emitted, {
            "raw_count": len(raw_sorted),
            "emitted_count": len(emitted),
            "duplicates_merged": merged_dupes,
            "reopen_blocked": reopen_blocks,
            "episodes": len(episodes),
        }

    def consolidate_by_video(
        self, by_video: dict[str, list[RawEvent]]
    ) -> tuple[dict[str, list[RawEvent]], dict[str, dict[str, int]], dict[str, int]]:
        out: dict[str, list[RawEvent]] = {}
        per_video: dict[str, dict[str, int]] = {}
        totals = {"raw_count": 0, "emitted_count": 0, "duplicates_merged": 0, "reopen_blocked": 0, "episodes": 0}
        for vid, evs in by_video.items():
            emitted, st = self.consolidate_video(evs)
            out[vid] = emitted
            per_video[vid] = st
            for k in totals:
                totals[k] += st[k]
        return out, per_video, totals


# ============================================================================
# V3: Multi-bbox anchor with episode-history matching + multi-person guard
# ============================================================================


@dataclass
class ConsolidatorV3Params:
    """V3 fixes the V2 order-of-processing bug:
    the anchor keeps a list of all (bbox, ts) within same_episode_max_gap_seconds.
    A new event matches the anchor if any historical bbox passes the spatial test.

    keep_multi_person_separation: if True, when the new event lies clearly OUTSIDE
    every historical bbox AND outside the expanded hull, it is NOT merged
    (keeps multi-person events separate).
    """

    same_episode_max_gap_seconds: float = 8.0
    expanded_anchor_bbox_ratio: float = 2.0
    center_distance_ratio: float = 0.65
    iou_threshold: float = 0.10
    history_seconds: float = 8.0
    keep_multi_person_separation: bool = True
    max_events_per_track_per_episode: int = 1
    allow_multiple_tracks_per_video: bool = True


class EventConsolidatorV3:
    def __init__(self, params: ConsolidatorV3Params | None = None) -> None:
        self.params = params or ConsolidatorV3Params()

    def _matches_history(self, history: list[tuple[float, tuple]], ev: RawEvent) -> bool:
        """Test event bbox against any historical bbox in the anchor."""
        params = self.params
        if ev.bbox is None:
            return False
        for _ts, hb in history:
            if hb is None:
                continue
            if _iou(ev.bbox, hb) >= params.iou_threshold:
                return True
            expanded = _expand_bbox(hb, params.expanded_anchor_bbox_ratio)
            ev_center = _bbox_center(ev.bbox)
            if _center_inside(expanded, ev_center):
                return True
            anchor_center = _bbox_center(hb)
            diag = max(_diag(hb), 1.0)
            dist = ((ev_center[0] - anchor_center[0]) ** 2 + (ev_center[1] - anchor_center[1]) ** 2) ** 0.5
            if dist / diag <= params.center_distance_ratio:
                return True
        return False

    def _anchor_matches(self, anchor: dict, ev: RawEvent) -> bool:
        params = self.params
        if ev.event_type != anchor["event_type"]:
            return False
        gap = ev.timestamp_sec - anchor["last_ts"]
        if gap < 0 or gap > params.same_episode_max_gap_seconds:
            return False
        if ev.person_track_id is not None and ev.person_track_id in anchor["track_ids"]:
            return True
        return self._matches_history(anchor["history"], ev)

    def consolidate_video(self, raw: list[RawEvent]) -> tuple[list[RawEvent], dict[str, int]]:
        params = self.params
        raw_sorted = sorted(raw, key=lambda r: (r.frame_idx, r.event_id))
        anchors: list[dict] = []
        emitted: list[RawEvent] = []
        merged_dupes = 0

        for ev in raw_sorted:
            # Trim per-anchor history older than history_seconds
            for a in anchors:
                a["history"] = [
                    (t, b) for (t, b) in a["history"] if ev.timestamp_sec - t <= params.history_seconds
                ]
            # Expire whole anchors past gap window
            anchors = [
                a for a in anchors
                if ev.timestamp_sec - a["last_ts"] <= params.same_episode_max_gap_seconds
            ]
            # Find ALL matching anchors (could be multiple if regions overlap)
            matched_anchors = [a for a in anchors if self._anchor_matches(a, ev)]
            if matched_anchors:
                # Pick the spatially closest one to avoid cross-person bleed
                if ev.bbox is not None:
                    def _score(a: dict) -> float:
                        ev_c = _bbox_center(ev.bbox)
                        best = float("inf")
                        for _ts, hb in a["history"]:
                            if hb is None:
                                continue
                            hc = _bbox_center(hb)
                            d = ((ev_c[0] - hc[0]) ** 2 + (ev_c[1] - hc[1]) ** 2) ** 0.5
                            if d < best:
                                best = d
                        return best
                    matched = min(matched_anchors, key=_score)
                else:
                    matched = matched_anchors[0]
                if ev.person_track_id is not None:
                    matched["track_ids"].add(ev.person_track_id)
                if ev.bbox is not None:
                    matched["history"].append((ev.timestamp_sec, ev.bbox))
                matched["last_frame"] = ev.frame_idx
                matched["last_ts"] = ev.timestamp_sec
                matched["count"] += 1
                merged_dupes += 1
                continue
            # No match → keep_multi_person_separation default (True) allows new anchor freely.
            anchors.append(
                {
                    "event_type": ev.event_type,
                    "track_ids": {ev.person_track_id} if ev.person_track_id is not None else set(),
                    "history": [(ev.timestamp_sec, ev.bbox)] if ev.bbox is not None else [],
                    "last_frame": ev.frame_idx,
                    "last_ts": ev.timestamp_sec,
                    "opened_ts": ev.timestamp_sec,
                    "count": 1,
                }
            )
            emitted.append(ev)

        return emitted, {
            "raw_count": len(raw_sorted),
            "emitted_count": len(emitted),
            "duplicates_merged": merged_dupes,
            "reopen_blocked": 0,
            "episodes": len(emitted),
        }

    def consolidate_by_video(
        self, by_video: dict[str, list[RawEvent]]
    ) -> tuple[dict[str, list[RawEvent]], dict[str, dict[str, int]], dict[str, int]]:
        out: dict[str, list[RawEvent]] = {}
        per_video: dict[str, dict[str, int]] = {}
        totals = {"raw_count": 0, "emitted_count": 0, "duplicates_merged": 0, "reopen_blocked": 0, "episodes": 0}
        for vid, evs in by_video.items():
            emitted, st = self.consolidate_video(evs)
            out[vid] = emitted
            per_video[vid] = st
            for k in totals:
                totals[k] += st[k]
        return out, per_video, totals


# ============================================================================
# V2: Anchor-based consolidator (better track-switch handling)
# ============================================================================


@dataclass
class ConsolidatorV2Params:
    """Anchor-based deduplication.

    Each open anchor stores the last bbox / center / event_type / track_ids /
    last_seen_ts. A new event is merged into an anchor if event_type matches
    AND time_gap <= same_zone_merge_seconds AND any of:
      - same track_id;
      - IoU(event.bbox, anchor.last_bbox) >= iou_threshold;
      - event center inside expanded(anchor.last_bbox, expanded_bbox_ratio);
      - center distance / anchor diag <= trajectory_center_ratio.
    An anchor expires after anchor_ttl_seconds since last_seen.
    """

    anchor_ttl_seconds: float = 8.0
    same_zone_merge_seconds: float = 8.0
    expanded_bbox_ratio: float = 1.5
    trajectory_center_ratio: float = 0.55
    iou_threshold: float = 0.10
    max_events_per_track_per_episode: int = 1
    allow_multiple_tracks_per_video: bool = True


def _expand_bbox(b: tuple[float, float, float, float], ratio: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    dx, dy = (ratio - 1.0) * w / 2.0, (ratio - 1.0) * h / 2.0
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def _center_inside(b: tuple[float, float, float, float], c: tuple[float, float]) -> bool:
    return b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]


def _diag(b: tuple[float, float, float, float]) -> float:
    return ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5


def _bbox_center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


class EventConsolidatorV2:
    def __init__(self, params: ConsolidatorV2Params | None = None) -> None:
        self.params = params or ConsolidatorV2Params()

    def _anchor_matches(self, anchor: dict, ev: RawEvent, params: ConsolidatorV2Params) -> bool:
        if ev.event_type != anchor["event_type"]:
            return False
        gap = ev.timestamp_sec - anchor["last_ts"]
        if gap < 0:
            return False
        # Anchor expires after TTL
        if gap > params.anchor_ttl_seconds:
            return False
        # Same-zone merge window: stricter than TTL — anchors stay alive but only merge while still close in time
        if gap > params.same_zone_merge_seconds:
            return False
        if ev.person_track_id is not None and ev.person_track_id in anchor["track_ids"]:
            return True
        if ev.bbox is None or anchor["last_bbox"] is None:
            return False
        if _iou(ev.bbox, anchor["last_bbox"]) >= params.iou_threshold:
            return True
        expanded = _expand_bbox(anchor["last_bbox"], params.expanded_bbox_ratio)
        ev_center = _bbox_center(ev.bbox)
        if _center_inside(expanded, ev_center):
            return True
        anchor_center = _bbox_center(anchor["last_bbox"])
        diag = max(_diag(anchor["last_bbox"]), 1.0)
        dist = ((ev_center[0] - anchor_center[0]) ** 2 + (ev_center[1] - anchor_center[1]) ** 2) ** 0.5
        if dist / diag <= params.trajectory_center_ratio:
            return True
        return False

    def consolidate_video(self, raw: list[RawEvent]) -> tuple[list[RawEvent], dict[str, int]]:
        params = self.params
        raw_sorted = sorted(raw, key=lambda r: (r.frame_idx, r.event_id))
        anchors: list[dict] = []
        emitted: list[RawEvent] = []
        merged_dupes = 0

        for ev in raw_sorted:
            anchors = [a for a in anchors if ev.timestamp_sec - a["last_ts"] <= params.anchor_ttl_seconds]
            matched = None
            for a in anchors:
                if self._anchor_matches(a, ev, params):
                    matched = a
                    break
            if matched is not None:
                if ev.person_track_id is not None:
                    matched["track_ids"].add(ev.person_track_id)
                if ev.bbox is not None:
                    matched["last_bbox"] = ev.bbox
                matched["last_frame"] = ev.frame_idx
                matched["last_ts"] = ev.timestamp_sec
                matched["count"] += 1
                merged_dupes += 1
                continue
            anchors.append(
                {
                    "event_type": ev.event_type,
                    "track_ids": {ev.person_track_id} if ev.person_track_id is not None else set(),
                    "last_bbox": ev.bbox,
                    "last_frame": ev.frame_idx,
                    "last_ts": ev.timestamp_sec,
                    "opened_ts": ev.timestamp_sec,
                    "count": 1,
                }
            )
            emitted.append(ev)

        return emitted, {
            "raw_count": len(raw_sorted),
            "emitted_count": len(emitted),
            "duplicates_merged": merged_dupes,
            "reopen_blocked": 0,
            "episodes": len(emitted),
        }

    def consolidate_by_video(
        self, by_video: dict[str, list[RawEvent]]
    ) -> tuple[dict[str, list[RawEvent]], dict[str, dict[str, int]], dict[str, int]]:
        out: dict[str, list[RawEvent]] = {}
        per_video: dict[str, dict[str, int]] = {}
        totals = {"raw_count": 0, "emitted_count": 0, "duplicates_merged": 0, "reopen_blocked": 0, "episodes": 0}
        for vid, evs in by_video.items():
            emitted, st = self.consolidate_video(evs)
            out[vid] = emitted
            per_video[vid] = st
            for k in totals:
                totals[k] += st[k]
        return out, per_video, totals
