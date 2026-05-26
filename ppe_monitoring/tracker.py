from __future__ import annotations

from collections import defaultdict, deque

from .geometry import (
    bbox_iou,
    box_center,
    clip_box_to_box,
    clip_xyxy_to_frame,
    is_box_touching_roi_edge,
)
from .types import Detection


class PersonTracker:
    def __init__(self, cfg: dict, class_ids: dict[str, int]):
        self.cfg = cfg
        self.class_ids = class_ids
        self.person_filter_cfg = cfg["filters"]["person"]
        self.head_hardhat_filter_cfg = cfg["filters"]["head_hardhat"]
        self.head_vs_person_cfg = cfg["filters"]["head_vs_person"]
        self.tracking_cfg = cfg["tracking"]
        self.roi_cfg = cfg["roi"]
        self.person_roi_cfg = cfg["person_roi"]
        self.model_cfg = cfg["model"]

        history_len = int(self.tracking_cfg.get("history_len", 10))
        self.history_xywh = defaultdict(lambda: deque(maxlen=history_len))
        self.history_xyxy = defaultdict(lambda: deque(maxlen=history_len))
        self.last_seen_frame: dict[int, int] = {}
        self.last_box_xyxy: dict[int, tuple[float, float, float, float]] = {}
        self.last_infer_person_ids: list[int] = []
        self.last_person_conf: dict[int, float] = {}
        self.last_person_filter_stats: dict[str, int] = {}
        self.last_head_hat_filter_stats: dict[str, int] = {}
        self.huge_bbox_hits: dict[int, int] = {}
        self.track_hits: dict[int, int] = {}
        self.track_headlike_hits: dict[int, int] = {}
        self.track_no_headlike_streak: dict[int, int] = {}
        self.synthetic_track_id = -1

    def _new_synthetic_track_id(self) -> int:
        tid = self.synthetic_track_id
        self.synthetic_track_id -= 1
        return tid

    def _is_valid_person_box(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        conf: float,
        frame_w: int,
        frame_h: int,
        roi_rect: tuple[int, int, int, int],
    ) -> tuple[bool, str]:
        x1, y1, x2, y2 = bbox_xyxy
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        area_ratio = (w * h) / float(frame_w * frame_h)
        aspect_ratio = w / max(h, 1e-6)
        width_ratio = w / float(frame_w)
        height_ratio = h / float(frame_h)
        in_roi = True
        if self.roi_cfg["enabled"] and self.roi_cfg.get("person_center_must_be_in_roi", True):
            rx1, ry1, rx2, ry2 = roi_rect
            in_roi = rx1 <= x <= rx2 and ry1 <= y <= ry2
        if conf < self.model_cfg["person_min_conf"]:
            return False, "low_conf"
        if not in_roi:
            return False, "outside_roi"
        if area_ratio < self.person_filter_cfg["min_area_ratio"]:
            return False, "small_area"
        if area_ratio > self.person_filter_cfg["max_area_ratio"]:
            return False, "large_area"
        if aspect_ratio < self.person_filter_cfg["min_aspect_ratio"]:
            return False, "bad_aspect_low"
        if aspect_ratio > self.person_filter_cfg["max_aspect_ratio"]:
            return False, "bad_aspect_high"
        if width_ratio > self.person_filter_cfg["max_width_ratio"]:
            return False, "too_wide"
        if height_ratio < self.person_filter_cfg["min_height_ratio"]:
            return False, "too_short"
        if height_ratio > self.person_filter_cfg["max_height_ratio"]:
            return False, "too_tall"
        return True, "accepted"

    def _is_valid_head_hardhat_box(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        x1, y1, x2, y2 = bbox_xyxy
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        area_ratio = (w * h) / float(frame_w * frame_h)
        aspect_ratio = w / h
        width_ratio = w / float(frame_w)
        height_ratio = h / float(frame_h)
        return (
            self.head_hardhat_filter_cfg["min_area_ratio"] <= area_ratio <= self.head_hardhat_filter_cfg["max_area_ratio"]
            and self.head_hardhat_filter_cfg["min_aspect_ratio"] <= aspect_ratio <= self.head_hardhat_filter_cfg["max_aspect_ratio"]
            and width_ratio <= self.head_hardhat_filter_cfg["max_width_ratio"]
            and height_ratio <= self.head_hardhat_filter_cfg["max_height_ratio"]
        )

    def _is_headlike_size_reasonable_for_person(
        self,
        headlike_box: tuple[float, float, float, float],
        person_box: tuple[float, float, float, float],
    ) -> bool:
        hx1, hy1, hx2, hy2 = headlike_box
        px1, py1, px2, py2 = person_box
        hw = max(1e-6, hx2 - hx1)
        hh = max(1e-6, hy2 - hy1)
        pw = max(1e-6, px2 - px1)
        ph = max(1e-6, py2 - py1)
        head_area = hw * hh
        person_area = pw * ph
        return (
            (head_area / person_area) <= self.head_vs_person_cfg["max_area_ratio_of_person"]
            and (hw / pw) <= self.head_vs_person_cfg["max_width_ratio_of_person"]
            and (hh / ph) <= self.head_vs_person_cfg["max_height_ratio_of_person"]
        )

    def _stabilize_person_box(
        self,
        prev_box: tuple[float, float, float, float],
        curr_box: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        px1, py1, px2, py2 = prev_box
        cx1, cy1, cx2, cy2 = curr_box
        prev_w = max(1e-6, px2 - px1)
        prev_h = max(1e-6, py2 - py1)
        curr_w = max(1e-6, cx2 - cx1)
        curr_h = max(1e-6, cy2 - cy1)
        prev_area = prev_w * prev_h
        curr_area = curr_w * curr_h
        prev_cx, prev_cy = box_center(prev_box)
        curr_cx, curr_cy = box_center(curr_box)

        grew_too_much = (
            curr_area > prev_area * self.tracking_cfg["max_area_growth_ratio"]
            or curr_w > prev_w * self.tracking_cfg["max_width_growth_ratio"]
            or curr_h > prev_h * self.tracking_cfg["max_height_growth_ratio"]
        )
        shifted_too_far = (
            abs(curr_cx - prev_cx) > (prev_w * float(self.tracking_cfg.get("max_center_shift_ratio_w", 0.65)))
            or abs(curr_cy - prev_cy) > (prev_h * float(self.tracking_cfg.get("max_center_shift_ratio_h", 0.50)))
        )
        if (grew_too_much or shifted_too_far) and bbox_iou(prev_box, curr_box) < self.tracking_cfg["min_iou_if_growth_triggered"]:
            return prev_box

        alpha = float(self.tracking_cfg.get("blend_alpha", 1.0))
        alpha = max(0.0, min(1.0, alpha))
        if alpha >= 0.999:
            return curr_box
        if alpha <= 0.001:
            return prev_box
        return (
            px1 * (1.0 - alpha) + cx1 * alpha,
            py1 * (1.0 - alpha) + cy1 * alpha,
            px2 * (1.0 - alpha) + cx2 * alpha,
            py2 * (1.0 - alpha) + cy2 * alpha,
        )

    @staticmethod
    def _expand_box_xyxy_for_match(
        box: tuple[float, float, float, float],
        expand_ratio: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float, float, float]:
        """Inflate box for association under occlusion (partial overlap with foreground objects)."""
        x1, y1, x2, y2 = box
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        pad_x = w * expand_ratio * 0.5
        pad_y = h * expand_ratio * 0.5
        nx1 = max(0.0, x1 - pad_x)
        ny1 = max(0.0, y1 - pad_y)
        nx2 = min(float(frame_w), x2 + pad_x)
        ny2 = min(float(frame_h), y2 + pad_y)
        return (nx1, ny1, nx2, ny2)

    def _match_existing_person_id(
        self,
        candidate_box: tuple[float, float, float, float],
        used_ids: set[int],
        frame_idx: int,
        frame_w: int,
        frame_h: int,
        single_person_frame: bool = False,
    ) -> int:
        best_id = None
        best_iou = 0.0
        best_center_ok = False
        max_gap = int(self.tracking_cfg["max_transfer_gap_frames"])
        iou_th = float(self.tracking_cfg["transfer_iou_threshold"])
        transfer_max_cx = float(self.tracking_cfg.get("transfer_max_center_shift_ratio_w", 0.55))
        transfer_max_cy = float(self.tracking_cfg.get("transfer_max_center_shift_ratio_h", 0.45))
        single_scale = float(self.tracking_cfg.get("transfer_single_person_center_scale", 1.35))
        if single_person_frame:
            transfer_max_cx *= single_scale
            transfer_max_cy *= single_scale
        min_iou_for_center_fallback = float(self.tracking_cfg.get("transfer_min_iou_for_center_fallback", 0.0))
        max_area_ratio_jump = float(self.tracking_cfg.get("transfer_max_area_ratio_jump", 2.2))
        require_area = bool(self.tracking_cfg.get("transfer_require_area_consistency", True)) and not single_person_frame
        expand_ratio = float(self.tracking_cfg.get("transfer_match_expand_ratio", 0.25))
        cx1, cy1 = box_center(candidate_box)
        cw = max(1e-6, candidate_box[2] - candidate_box[0])
        ch = max(1e-6, candidate_box[3] - candidate_box[1])
        c_area = cw * ch
        for old_id, old_box in self.last_box_xyxy.items():
            if old_id in used_ids:
                continue
            if (frame_idx - self.last_seen_frame.get(old_id, -10**9)) > max_gap:
                continue
            old_expanded = self._expand_box_xyxy_for_match(old_box, expand_ratio, frame_w, frame_h)
            iou = max(
                bbox_iou(candidate_box, old_box),
                bbox_iou(candidate_box, old_expanded),
            )
            ox1, oy1 = box_center(old_box)
            ow = max(1e-6, old_box[2] - old_box[0])
            oh = max(1e-6, old_box[3] - old_box[1])
            o_area = ow * oh
            shift_w = abs(cx1 - ox1) / max(ow, cw)
            shift_h = abs(cy1 - oy1) / max(oh, ch)
            area_jump = c_area / max(1e-6, o_area)
            area_ok = (
                area_jump <= max_area_ratio_jump
                and (1.0 / max_area_ratio_jump) <= area_jump
            )
            center_ok = shift_w <= transfer_max_cx and shift_h <= transfer_max_cy
            if require_area:
                center_ok = center_ok and area_ok
            if iou > best_iou:
                best_iou = iou
                best_id = old_id
                best_center_ok = center_ok
            elif iou == best_iou and center_ok and not best_center_ok:
                best_id = old_id
                best_center_ok = center_ok
        if best_id is None:
            return self._new_synthetic_track_id()
        if best_iou >= iou_th:
            return int(best_id)
        # Fast motion: IoU can drop to ~0 while center stays close to previous box.
        if best_center_ok and best_iou >= min_iou_for_center_fallback:
            return int(best_id)
        return self._new_synthetic_track_id()

    def _resolve_track_id(
        self,
        det_track_id: int | None,
        candidate_box: tuple[float, float, float, float],
        used_ids: set[int],
        frame_idx: int,
        frame_w: int,
        frame_h: int,
        single_person_frame: bool = False,
    ) -> int:
        """Resolve stable person id and recover from detector ID switches."""
        if det_track_id is None:
            return self._match_existing_person_id(
                candidate_box, used_ids, frame_idx, frame_w, frame_h, single_person_frame
            )

        incoming_id = int(det_track_id)
        incoming_recent = (frame_idx - self.last_seen_frame.get(incoming_id, -10**9)) <= int(
            self.tracking_cfg["max_transfer_gap_frames"]
        )
        if incoming_recent and incoming_id not in used_ids:
            return incoming_id

        # If detector emitted a new/unstable id, try to continue the closest recent track.
        matched_existing = self._match_existing_person_id(
            candidate_box, used_ids, frame_idx, frame_w, frame_h, single_person_frame
        )
        if matched_existing < 0:
            # _match_existing_person_id returns synthetic negative id when no match.
            if incoming_id not in used_ids:
                return incoming_id
            return matched_existing
        return matched_existing

    def update_person_tracks(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int, int],
        roi_rect: tuple[int, int, int, int],
        frame_idx: int,
    ) -> dict[int, tuple[float, float, float, float]]:
        frame_h, frame_w = frame_shape[:2]
        person_boxes: dict[int, tuple[float, float, float, float]] = {}
        used_ids: set[int] = set()
        stats: dict[str, int] = defaultdict(int)

        valid_entries: list[tuple[Detection, tuple[float, float, float, float]]] = []
        for det in detections:
            if det.cls_name != "person":
                continue
            stats["raw_person_detections"] += 1
            x1, y1, x2, y2 = clip_xyxy_to_frame(*det.bbox_xyxy, frame_w, frame_h)
            box = (x1, y1, x2, y2)

            if (
                self.roi_cfg["enabled"]
                and self.roi_cfg.get("global_inference_in_roi", False)
                and self.roi_cfg.get("reject_person_boxes_on_roi_edge", True)
                and is_box_touching_roi_edge(
                    box,
                    roi_rect,
                    int(self.roi_cfg.get("person_edge_reject_margin_px", 8)),
                )
            ):
                stats["dropped_roi_edge"] += 1
                continue
            if det.source == "binary_hardhat":
                if det.conf < float(self.model_cfg.get("person_min_conf", 0.35)):
                    stats["dropped_low_conf_binary"] += 1
                    continue
            else:
                valid, reason = self._is_valid_person_box(box, det.conf, frame_w, frame_h, roi_rect)
                if not valid:
                    stats[f"dropped_{reason}"] += 1
                    continue
            valid_entries.append((det, box))

        single_person_frame = len(valid_entries) == 1

        for det, box in valid_entries:
            track_id = self._resolve_track_id(
                det.track_id, box, used_ids, frame_idx, frame_w, frame_h, single_person_frame
            )
            if track_id in used_ids:
                track_id = self._new_synthetic_track_id()
            used_ids.add(track_id)

            prev_track = self.history_xyxy[track_id]
            prev_box = prev_track[-1] if len(prev_track) > 0 else None
            box, accepted = self._apply_person_bbox_sanity(
                track_id=track_id,
                curr_box=box,
                prev_box=prev_box,
                frame_w=frame_w,
                frame_h=frame_h,
                stats=stats,
            )
            if not accepted:
                continue
            if prev_box is not None:
                box = self._stabilize_person_box(prev_box, box)

            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            self.history_xyxy[track_id].append(box)
            self.history_xywh[track_id].append((cx, cy, w, h))
            self.last_seen_frame[track_id] = frame_idx
            self.last_box_xyxy[track_id] = box
            self.last_person_conf[track_id] = float(det.conf)
            self.track_hits[track_id] = int(self.track_hits.get(track_id, 0)) + 1
            person_boxes[track_id] = box
            stats["accepted_person_tracks"] += 1

        self.last_infer_person_ids = list(person_boxes.keys())
        stats["final_person_tracks"] = len(person_boxes)
        self.last_person_filter_stats = dict(stats)
        return person_boxes

    def _apply_person_bbox_sanity(
        self,
        track_id: int,
        curr_box: tuple[float, float, float, float],
        prev_box: tuple[float, float, float, float] | None,
        frame_w: int,
        frame_h: int,
        stats: dict[str, int],
    ) -> tuple[tuple[float, float, float, float], bool]:
        x1, y1, x2, y2 = curr_box
        cw = max(1.0, x2 - x1)
        ch = max(1.0, y2 - y1)
        c_area = cw * ch
        c_area_ratio = c_area / float(frame_w * frame_h)
        max_area_ratio = float(self.tracking_cfg.get("max_person_area_ratio", 0.55))
        huge_confirm = int(self.tracking_cfg.get("huge_bbox_confirm_frames", 3))

        if c_area_ratio > max_area_ratio:
            self.huge_bbox_hits[track_id] = int(self.huge_bbox_hits.get(track_id, 0)) + 1
            if self.huge_bbox_hits[track_id] < huge_confirm:
                stats["rejected_person_huge_bbox"] += 1
                stats["rejected_huge_bbox_count"] += 1
                if prev_box is not None:
                    stats["reused_previous_bbox_due_to_jump"] += 1
                    return prev_box, True
                return curr_box, False
        else:
            self.huge_bbox_hits[track_id] = 0

        if prev_box is None:
            return curr_box, True

        px1, py1, px2, py2 = prev_box
        pw = max(1.0, px2 - px1)
        ph = max(1.0, py2 - py1)
        p_area = pw * ph
        max_area_jump = float(self.tracking_cfg.get("max_area_jump_ratio", 2.8))
        max_w_jump = float(self.tracking_cfg.get("max_width_jump_ratio", 2.5))
        max_h_jump = float(self.tracking_cfg.get("max_height_jump_ratio", 2.5))
        area_jump = c_area / max(1.0, p_area)
        width_jump = cw / pw
        height_jump = ch / ph
        if area_jump > max_area_jump or width_jump > max_w_jump or height_jump > max_h_jump:
            stats["rejected_person_area_jump"] += 1
            stats["rejected_jump_bbox_count"] += 1
            stats["reused_previous_bbox_due_to_jump"] += 1
            return prev_box, True
        return curr_box, True

    def get_active_person_boxes(
        self,
        frame_idx: int,
        max_gap_frames: int,
    ) -> dict[int, tuple[float, float, float, float]]:
        boxes: dict[int, tuple[float, float, float, float]] = {}
        for person_id in self.last_infer_person_ids:
            if person_id not in self.last_seen_frame:
                continue
            if (frame_idx - self.last_seen_frame[person_id]) > max_gap_frames:
                continue
            if len(self.history_xyxy[person_id]) == 0:
                continue
            boxes[person_id] = self.history_xyxy[person_id][-1]
        return boxes

    def prune_stale_tracks(self, frame_idx: int, max_stale_frames: int) -> list[int]:
        removed: list[int] = []
        for person_id, seen_frame in list(self.last_seen_frame.items()):
            if (frame_idx - seen_frame) <= max_stale_frames:
                continue
            removed.append(person_id)
            self.last_seen_frame.pop(person_id, None)
            self.last_box_xyxy.pop(person_id, None)
            self.history_xyxy.pop(person_id, None)
            self.history_xywh.pop(person_id, None)
            self.last_person_conf.pop(person_id, None)
            self.huge_bbox_hits.pop(person_id, None)
            self.track_hits.pop(person_id, None)
            self.track_headlike_hits.pop(person_id, None)
            self.track_no_headlike_streak.pop(person_id, None)
        if removed:
            removed_set = set(removed)
            self.last_infer_person_ids = [pid for pid in self.last_infer_person_ids if pid not in removed_set]
        return removed

    def _find_owner_person_for_headlike_box(
        self,
        candidate_box: tuple[float, float, float, float],
        person_boxes: dict[int, tuple[float, float, float, float]],
    ) -> int | None:
        cx, cy = box_center(candidate_box)
        best_person = None
        best_score = -1.0
        for person_id, pbox in person_boxes.items():
            px1, py1, px2, py2 = pbox
            if not (px1 <= cx <= px2 and py1 <= cy <= py2):
                continue
            person_h = py2 - py1
            max_head_y = py1 + self.person_roi_cfg["y_bottom_ratio"] * person_h
            if cy > max_head_y:
                continue
            p_cx, _ = box_center(pbox)
            p_w = max(1.0, px2 - px1)
            center_alignment = 1.0 - min(1.0, abs(cx - p_cx) / (0.5 * p_w))
            upper_bias = 1.0 - min(1.0, max(0.0, cy - py1) / max(1.0, max_head_y - py1))
            score = 0.35 * bbox_iou(candidate_box, pbox) + 0.40 * center_alignment + 0.25 * upper_bias
            if score > best_score:
                best_score = score
                best_person = person_id
        return best_person

    def associate_head_hardhat(
        self,
        detections: list[Detection],
        person_boxes: dict[int, tuple[float, float, float, float]],
        frame_shape: tuple[int, int, int],
    ) -> tuple[dict[int, bool], list[Detection], list[Detection]]:
        frame_h, frame_w = frame_shape[:2]
        person_hardhat_observed = {person_id: False for person_id in person_boxes}
        accepted_heads: list[Detection] = []
        accepted_hardhats: list[Detection] = []
        stats: dict[str, int] = defaultdict(int)

        for det in detections:
            if det.cls_name not in {"head", "hardhat"}:
                continue
            stats["raw_head_hat"] += 1
            x1, y1, x2, y2 = clip_xyxy_to_frame(*det.bbox_xyxy, frame_w, frame_h)
            box = (x1, y1, x2, y2)
            if not self._is_valid_head_hardhat_box(box, frame_w, frame_h):
                stats["rejected_head_hat_geom"] += 1
                continue

            owner_person = det.owner_person_id
            if owner_person is None and det.track_id is not None and det.track_id in person_boxes:
                owner_person = int(det.track_id)
            if owner_person is None:
                owner_person = self._find_owner_person_for_headlike_box(box, person_boxes)
            if owner_person is None or owner_person not in person_boxes:
                stats["rejected_head_hat_no_owner"] += 1
                continue

            clipped_box = clip_box_to_box(box, person_boxes[owner_person])
            if clipped_box is None:
                stats["rejected_head_hat_outside_person"] += 1
                continue
            if not self._is_headlike_size_reasonable_for_person(clipped_box, person_boxes[owner_person]):
                stats["rejected_head_hat_size"] += 1
                continue

            dedup_set = accepted_heads if det.cls_name == "head" else accepted_hardhats
            dedup_iou = float(self.tracking_cfg.get("headlike_dedup_iou", 0.45))
            if any(bbox_iou(clipped_box, old.bbox_xyxy) >= dedup_iou for old in dedup_set):
                stats["rejected_head_hat_dedup"] += 1
                continue

            accepted = Detection(
                cls_id=det.cls_id,
                cls_name=det.cls_name,
                conf=det.conf,
                bbox_xyxy=clipped_box,
                track_id=det.track_id,
                source=det.source,
                owner_person_id=owner_person,
            )
            if det.cls_name == "head":
                accepted_heads.append(accepted)
                stats["accepted_heads"] += 1
            else:
                accepted_hardhats.append(accepted)
                person_hardhat_observed[owner_person] = True
                stats["accepted_hardhats"] += 1

        self.last_head_hat_filter_stats = dict(stats)
        return person_hardhat_observed, accepted_heads, accepted_hardhats

    def register_headlike_support(
        self,
        person_boxes: dict[int, tuple[float, float, float, float]],
        accepted_heads: list[Detection],
        accepted_hardhats: list[Detection],
    ) -> None:
        seen_support: set[int] = set()
        for d in accepted_heads + accepted_hardhats:
            if d.owner_person_id is not None:
                seen_support.add(int(d.owner_person_id))
        for pid in person_boxes:
            if pid in seen_support:
                self.track_headlike_hits[pid] = int(self.track_headlike_hits.get(pid, 0)) + 1
                self.track_no_headlike_streak[pid] = 0
            else:
                self.track_no_headlike_streak[pid] = int(self.track_no_headlike_streak.get(pid, 0)) + 1

    def get_valid_person_ids(self, person_boxes: dict[int, tuple[float, float, float, float]]) -> set[int]:
        min_hits = int(self.tracking_cfg.get("min_track_hits_for_valid_person", 3))
        max_no_headlike = int(self.tracking_cfg.get("max_no_headlike_streak_frames", 8))
        hi_conf = float(self.tracking_cfg.get("valid_person_high_conf_override", 0.70))
        out: set[int] = set()
        for pid in person_boxes:
            hits = int(self.track_hits.get(pid, 0))
            if hits < min_hits:
                continue
            head_hits = int(self.track_headlike_hits.get(pid, 0))
            no_headlike_streak = int(self.track_no_headlike_streak.get(pid, 0))
            conf = float(self.last_person_conf.get(pid, 0.0))
            if head_hits <= 0 and no_headlike_streak > max_no_headlike and conf < hi_conf:
                continue
            out.add(pid)
        return out

