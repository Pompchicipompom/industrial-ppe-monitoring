from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

from .geometry import bbox_iou
from .types import Detection


def _expand_xyxy_for_spatial_match(
    box: tuple[float, float, float, float],
    expand_ratio: float,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    pad_x = w * expand_ratio * 0.5
    pad_y = h * expand_ratio * 0.5
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(frame_w), x2 + pad_x),
        min(float(frame_h), y2 + pad_y),
    )


def _bgr_lerp(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    return tuple(int(round(x * (1.0 - t) + y * t)) for x, y in zip(a, b))


def _draw_dashed_rectangle(
    frame,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int = 1,
    dash: int = 8,
    gap: int = 5,
) -> None:
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    step = dash + gap
    for x in range(x1, x2, step):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
    for x in range(x1, x2, step):
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, step):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
    for y in range(y1, y2, step):
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)


@dataclass
class _VisualTrack:
    bbox: tuple[float, float, float, float]
    ttl: int


class VisualizationRenderer:
    def __init__(self, vis_cfg: dict):
        self.vis_cfg = vis_cfg
        self.head_tracks: dict[str, _VisualTrack] = {}
        self.hardhat_tracks: dict[str, _VisualTrack] = {}
        self.person_hardhat_locked: set[int] = set()
        self.person_violation_locked: set[int] = set()
        self.person_last_seen_frame: dict[int, int] = {}
        self.skipped_stale_visual_bbox = 0
        # Recent bbox snapshots for violators — survives Ultralytics track_id switches.
        self._violation_anchors: list[tuple[tuple[float, float, float, float], int]] = []
        # «Внимание» остаётся после первого нарушения / события (пока идёт ролик).
        self.panel_attention_latched: bool = False

    def _refresh_violation_anchors(
        self,
        person_boxes: dict[int, tuple[float, float, float, float]],
        violating_person_ids: set[int],
        frame_idx: int,
    ) -> None:
        if not bool(self.vis_cfg.get("violation_spatial_inherit_enabled", True)):
            return
        ttl = int(self.vis_cfg.get("violation_spatial_anchor_ttl_frames", 150))
        max_a = int(self.vis_cfg.get("violation_spatial_max_anchors", 50))
        for pid in violating_person_ids:
            if pid in person_boxes:
                self._violation_anchors.append((person_boxes[pid], frame_idx))
        for pid in self.person_violation_locked:
            if pid in person_boxes:
                self._violation_anchors.append((person_boxes[pid], frame_idx))
        self._violation_anchors = [(b, f) for b, f in self._violation_anchors if frame_idx - f <= ttl]
        if len(self._violation_anchors) > max_a:
            self._violation_anchors = self._violation_anchors[-max_a:]

    def _spatial_inherits_violation(
        self,
        box: tuple[float, float, float, float],
        frame_idx: int,
        frame_w: int,
        frame_h: int,
    ) -> bool:
        if not bool(self.vis_cfg.get("violation_spatial_inherit_enabled", True)):
            return False
        ttl = int(self.vis_cfg.get("violation_spatial_anchor_ttl_frames", 150))
        iou_min = float(self.vis_cfg.get("violation_spatial_iou_min", 0.12))
        expand = float(self.vis_cfg.get("violation_spatial_expand_ratio", 0.35))
        for ab, af in self._violation_anchors:
            if frame_idx - af > ttl:
                continue
            a_exp = _expand_xyxy_for_spatial_match(ab, expand, frame_w, frame_h)
            if max(bbox_iou(box, ab), bbox_iou(box, a_exp)) >= iou_min:
                return True
        return False

    def _smooth_headlike(
        self,
        detections: list[Detection],
        tracks: dict[str, _VisualTrack],
        frame_idx: int,
    ) -> list[Detection]:
        enabled = bool(self.vis_cfg.get("smoothing_enabled", True))
        if not enabled:
            return detections
        alpha = float(self.vis_cfg.get("bbox_smoothing_alpha", 0.4))
        alpha = max(0.05, min(0.95, alpha))
        ttl = int(self.vis_cfg.get("head_hat_ttl_frames", self.vis_cfg.get("bbox_ttl_frames", 3)))
        iou_th = float(self.vis_cfg.get("bbox_smoothing_match_iou", 0.20))
        used: set[str] = set()
        out: list[Detection] = []

        for det in detections:
            key = f"f{frame_idx}-{len(out)}"
            best_key = None
            best_iou = 0.0
            for old_key, tr in tracks.items():
                if old_key in used:
                    continue
                iou = bbox_iou(det.bbox_xyxy, tr.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_key = old_key
            if best_key is not None and best_iou >= iou_th:
                prev = tracks[best_key].bbox
                x1 = prev[0] * (1.0 - alpha) + det.bbox_xyxy[0] * alpha
                y1 = prev[1] * (1.0 - alpha) + det.bbox_xyxy[1] * alpha
                x2 = prev[2] * (1.0 - alpha) + det.bbox_xyxy[2] * alpha
                y2 = prev[3] * (1.0 - alpha) + det.bbox_xyxy[3] * alpha
                smooth_box = (x1, y1, x2, y2)
                tracks[best_key] = _VisualTrack(bbox=smooth_box, ttl=ttl)
                used.add(best_key)
                out.append(
                    Detection(
                        cls_id=det.cls_id,
                        cls_name=det.cls_name,
                        conf=det.conf,
                        bbox_xyxy=smooth_box,
                        track_id=det.track_id,
                        source=det.source,
                        owner_person_id=det.owner_person_id,
                        hardhat_state=det.hardhat_state,
                    )
                )
            else:
                tracks[key] = _VisualTrack(bbox=det.bbox_xyxy, ttl=ttl)
                used.add(key)
                out.append(det)

        for key in list(tracks.keys()):
            if key in used:
                continue
            tracks[key].ttl -= 1
            if tracks[key].ttl <= 0:
                tracks.pop(key, None)
        return out

    def draw_runtime_overlay(
        self,
        frame,
        last_main_detections: list[Detection],
        accepted_heads: list[Detection],
        accepted_hardhats: list[Detection],
        person_boxes: dict[int, tuple[float, float, float, float]],
        confirmed_ids_for_draw: set[int],
        statuses: dict[int, bool],
        event_logic,
        violating_person_ids: set[int],
        frame_idx: int,
        input_fps: float,
        person_confirm_mode: str,
        active_violations: int,
        total_events: int,
        person_hardhat_observed: dict[int, bool],
        active_no_hardhat_now: int,
        unique_no_hardhat_total: int,
    ) -> None:
        vis_cfg = self.vis_cfg
        smoothed_heads = self._smooth_headlike(accepted_heads, self.head_tracks, frame_idx)
        smoothed_hardhats = self._smooth_headlike(accepted_hardhats, self.hardhat_tracks, frame_idx)
        draw_raw = bool(vis_cfg.get("draw_raw_detections", vis_cfg.get("draw_all_main_detections", False)))
        color_person = tuple(vis_cfg.get("person_color", (255, 0, 0)))
        color_head = tuple(vis_cfg.get("head_color", (0, 255, 255)))
        color_hardhat = tuple(vis_cfg.get("hardhat_color", (0, 255, 0)))
        color_vest = tuple(vis_cfg.get("vest_color", (255, 255, 0)))
        color_no_vest = tuple(vis_cfg.get("no_vest_color", (0, 165, 255)))
        color_violation = tuple(vis_cfg.get("violation_color", (0, 0, 255)))
        color_text = tuple(vis_cfg.get("text_color", (255, 255, 255)))
        thickness = int(vis_cfg.get("bbox_thickness", 2))
        label_scale = 0.52
        color_person_neutral = color_person
        color_person_with_hardhat = color_hardhat
        person_ttl = int(vis_cfg.get("person_ttl_frames", 12))

        for pid in person_boxes:
            self.person_last_seen_frame[pid] = frame_idx
        for pid in violating_person_ids:
            self.person_violation_locked.add(pid)
            self.person_last_seen_frame[pid] = frame_idx
        frame_h, frame_w = frame.shape[:2]
        self._refresh_violation_anchors(person_boxes, violating_person_ids, frame_idx)
        for pid in list(self.person_hardhat_locked):
            if (frame_idx - self.person_last_seen_frame.get(pid, -10**9)) > person_ttl:
                self.person_hardhat_locked.discard(pid)
                self.person_last_seen_frame.pop(pid, None)

        if draw_raw and last_main_detections:
            show_conf = vis_cfg.get("draw_detection_confidence_labels", False)
            skip_raw = {"person", "head", "hardhat"}
            default_color = (180, 180, 180)
            for d in last_main_detections:
                if d.cls_name in skip_raw:
                    continue
                x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
                cv2.rectangle(frame, (x1, y1), (x2, y2), default_color, 1)
                label = f"{d.cls_name} {d.conf:.2f}" if show_conf else d.cls_name
                cv2.putText(frame, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, default_color, 1)

        if vis_cfg.get("draw_head_hardhat_boxes", True):
            for det in smoothed_heads:
                x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_head, thickness)
                cv2.putText(frame, "head", (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_head, 1)
            for det in smoothed_hardhats:
                x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_hardhat, thickness)
                cv2.putText(frame, "hardhat", (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_hardhat, 1)
        if vis_cfg.get("draw_vest_boxes", True):
            for det in last_main_detections:
                if det.cls_name != "vest":
                    continue
                x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_vest, 1)
                cv2.putText(frame, "vest", (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_vest, 1)
        if vis_cfg.get("draw_no_vest_boxes", False):
            for det in last_main_detections:
                if det.cls_name != "no_vest":
                    continue
                x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_no_vest, 1)
                cv2.putText(frame, "no_vest", (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_no_vest, 1)

        draw_tracker_boxes = bool(vis_cfg.get("draw_tracker_boxes", vis_cfg.get("draw_person_boxes", True)))
        if draw_tracker_boxes:
            show_weak = bool(vis_cfg.get("show_weak_person", False))
            for person_id, (x1, y1, x2, y2) in person_boxes.items():
                is_confirmed = person_id in confirmed_ids_for_draw
                if person_confirm_mode == "soft" and not is_confirmed and not show_weak:
                    continue
                xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
                if not is_confirmed:
                    weak_col = (0, 200, 255)
                    _draw_dashed_rectangle(frame, xi1, yi1, xi2, yi2, weak_col, thickness=thickness)
                    cv2.putText(
                        frame, f"weak | id:{person_id}", (xi1, max(12, yi1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, weak_col, 1
                    )
                    continue

                if person_id not in self.person_violation_locked and self._spatial_inherits_violation(
                    (x1, y1, x2, y2), frame_idx, frame_w, frame_h
                ):
                    self.person_violation_locked.add(person_id)
                has_hardhat = bool(statuses.get(person_id, False))
                person_has_recorded_violation = person_id in self.person_violation_locked
                if has_hardhat and not person_has_recorded_violation:
                    person_color = color_person_with_hardhat
                elif person_id in violating_person_ids or person_has_recorded_violation:
                    person_color = color_violation
                else:
                    person_color = color_person_neutral
                seen_seconds = event_logic.seen_seconds(person_id, frame_idx, input_fps)
                person_status = "with hardhat" if has_hardhat else "person"
                cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), person_color, thickness)
                cv2.putText(
                    frame,
                    f"{person_status} | t: {seen_seconds:.1f}s | id:{person_id}",
                    (xi1, max(12, yi1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    label_scale,
                    person_color,
                    2,
                )

                if vis_cfg.get("show_violation_dot", True) and (
                    person_id in violating_person_ids or person_id in self.person_violation_locked
                ):
                    blink_period = max(2, int(vis_cfg.get("dot_blink_period_frames", 14)))
                    blink_on = (frame_idx % blink_period) < (blink_period // 2)
                    if blink_on:
                        dot_x = int((x1 + x2) / 2.0)
                        dot_y = max(8, int(y1) - 14)
                        dot_r = max(2, int(vis_cfg.get("dot_radius", 8)))
                        cv2.circle(frame, (dot_x, dot_y), dot_r + 2, (0, 0, 0), -1)
                        cv2.circle(frame, (dot_x, dot_y), dot_r, color_violation, -1)

        if vis_cfg.get("show_violation_banner", True):
            sticky = bool(vis_cfg.get("panel_attention_sticky", True))
            self.panel_attention_latched |= active_violations > 0 or int(total_events) > 0
            attention_active = self.panel_attention_latched if sticky else (active_violations > 0)
            _draw_status_panel(
                frame=frame,
                attention_active=attention_active,
                unique_no_hardhat_total=unique_no_hardhat_total,
                total_events=total_events,
                text_color=color_text,
                accent_color=color_violation,
                hide_when_idle=bool(vis_cfg.get("hide_panel_when_idle", False)),
                frame_idx=frame_idx,
                attention_text=str(vis_cfg.get("panel_attention_text", "Внимание")),
                blink_period_frames=int(
                    vis_cfg.get("panel_blink_period_frames", vis_cfg.get("dot_blink_period_frames", 14))
                ),
            )


def _draw_violation_banner(
    frame,
    text: str,
    top_left: tuple[int, int],
    bg_color_bgr: tuple[int, int, int],
    text_color_bgr: tuple[int, int, int],
) -> None:
    x, y = top_left
    if Image is None or ImageFont is None:
        cv2.rectangle(frame, (x, y), (x + 420, y + 38), bg_color_bgr, -1)
        cv2.putText(frame, text, (x + 8, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.78, text_color_bgr, 2)
        return

    font = _load_cyrillic_font(30)
    if font is None:
        cv2.rectangle(frame, (x, y), (x + 420, y + 38), bg_color_bgr, -1)
        cv2.putText(frame, text, (x + 8, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.78, text_color_bgr, 2)
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x = 10
    pad_y = 6
    draw.rectangle((x, y, x + tw + 2 * pad_x, y + th + 2 * pad_y), fill=_bgr_to_rgb(bg_color_bgr))
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=_bgr_to_rgb(text_color_bgr))
    frame[:, :, :] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _draw_status_panel(
    frame,
    attention_active: bool,
    unique_no_hardhat_total: int,
    total_events: int,
    text_color: tuple[int, int, int],
    accent_color: tuple[int, int, int],
    hide_when_idle: bool = False,
    frame_idx: int = 0,
    attention_text: str = "Внимание",
    blink_period_frames: int = 14,
) -> None:
    # hide_when_idle / unique_no_hardhat_total — совместимость вызова; счётчик показываем всегда (в т.ч. 0).
    _ = (hide_when_idle, unique_no_hardhat_total)
    alert = attention_active
    lines: list[str] = (
        [attention_text, f"Нарушений: {int(total_events)}"]
        if alert
        else [f"Нарушений: {int(total_events)}"]
    )
    blink_period = max(2, int(blink_period_frames))
    blink_on = (frame_idx % blink_period) < (blink_period // 2)
    attention_bgr = accent_color if blink_on else _bgr_lerp(accent_color, text_color, 0.55)

    panel_x = 12
    panel_y = 12
    pad = max(8, int(frame.shape[0] * 0.012))
    font_px = max(16, int(frame.shape[0] * 0.034))
    text_left_cv = panel_x + pad + 4
    line_step = max(22, int(font_px * 0.95))
    if Image is None or ImageFont is None:
        if alert and blink_on:
            ph = line_step * len(lines) + pad
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 6, panel_y + ph), accent_color, -1)
        y = panel_y + pad + int(font_px * 0.72)
        for i, line in enumerate(lines):
            col = attention_bgr if alert and i == 0 else text_color
            cv2.putText(frame, line, (text_left_cv, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            y += line_step
        return

    font = _load_cyrillic_font(font_px)
    if font is None:
        if alert and blink_on:
            ph = line_step * len(lines) + pad
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 6, panel_y + ph), accent_color, -1)
        y = panel_y + pad + int(font_px * 0.72)
        for i, line in enumerate(lines):
            col = attention_bgr if alert and i == 0 else text_color
            cv2.putText(frame, line, (text_left_cv, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            y += line_step
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    widths = []
    heights = []
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font)
        widths.append(bb[2] - bb[0])
        heights.append(bb[3] - bb[1])
    line_gap = max(4, int(font_px * 0.20))
    text_h = sum(heights) + (len(lines) - 1) * line_gap
    panel_w = max(widths) + 2 * pad
    panel_h = text_h + 2 * pad

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    panel_bg = (18, 18, 18, 170)
    od.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=8, fill=panel_bg)
    if alert and blink_on:
        ax = panel_x
        ay = panel_y
        od.rounded_rectangle((ax, ay, ax + 6, ay + panel_h), radius=3, fill=_bgr_to_rgb(accent_color) + (255,))

    ty = panel_y + pad
    text_left = panel_x + pad + 4
    for i, line in enumerate(lines):
        color_bgr = attention_bgr if alert and i == 0 else text_color
        od.text((text_left, ty), line, font=font, fill=_bgr_to_rgb(color_bgr) + (255,))
        ty += heights[i] + line_gap

    composed = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    frame[:, :, :] = cv2.cvtColor(np.array(composed), cv2.COLOR_RGB2BGR)


def _load_cyrillic_font(size: int):
    if ImageFont is None:
        return None
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            continue
    return None


def _bgr_to_rgb(color_bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = color_bgr
    return (r, g, b)
