from __future__ import annotations

"""
Industrial video-first PPE monitoring pipeline.

Architecture diagram and rationale: see [docs/architecture.md](../docs/architecture.md).
"""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .detector import PPEDetector
from .event_logic import TemporalEventLogic
from .geometry import get_roi_rect
from .metrics_constants import metrics_csv_fieldnames
from .motion import FrameSampler, InferenceGate, MotionDetector
from .person_confirmation import person_soft_confirmed, resolve_person_confirmation_mode
from .person_head_confirmation import person_confirmed_by_head_or_hardhat
from .profiler import RuntimeProfiler
from .rtsp_health import RtspHealthWatchdog
from .tracker import PersonTracker
from .types import Detection, ViolationEvent
from .video_id import resolve_video_id as _resolve_video_id
from .visualization import VisualizationRenderer

PRED_EVENT_SCHEMA_VERSION = 1
STAGE_TIMING_FIELDS = [
    "main_infer_ms",
    "roi_infer_ms",
    "tracking_ms",
    "event_logic_ms",
    "render_ms",
    "io_ms",
    "total_loop_ms",
]


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def create_video_writer(cap: cv2.VideoCapture, output_path: str, frame_size: tuple[int, int]):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    candidates = [
        (output_path, "mp4v"),
        ("output_files/processed.avi", "XVID"),
        ("output_files/processed.avi", "MJPG"),
    ]
    requested_resolved = str(Path(output_path).resolve())
    for out_path, fourcc_name in candidates:
        ensure_parent_dir(out_path)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
        if writer.isOpened():
            actual_resolved = str(Path(out_path).resolve())
            if actual_resolved != requested_resolved:
                print(
                    "WARNING: VideoWriter could not use configured path; writing elsewhere. "
                    f"configured={output_path!r} actual={out_path!r}"
                )
            print(f"Using output codec {fourcc_name} -> {out_path}")
            return writer, out_path, fps
        writer.release()

    raise RuntimeError("Could not initialize VideoWriter with available codecs.")


class VideoSource:
    def __init__(self, source, pipeline_cfg: dict):
        self.pipeline_cfg = pipeline_cfg
        self.source = self._normalize_source(source)
        self.is_rtsp = isinstance(self.source, str) and str(self.source).lower().startswith(("rtsp://", "rtsps://"))
        self.cap: cv2.VideoCapture | None = None
        self.health = RtspHealthWatchdog(is_rtsp=self.is_rtsp)

    @staticmethod
    def _normalize_source(source):
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    def open(self) -> bool:
        if self.is_rtsp:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.source)
        success = bool(self.cap is not None and self.cap.isOpened())
        self.health.on_open_result(success=success)
        return success

    def read(self):
        if self.cap is None:
            return False, None
        success, frame = self.cap.read()
        if success:
            self.health.on_frame_read_success()
            return True, frame
        self.health.on_frame_read_failure()
        if not self.is_rtsp:
            return False, None

        self.health.on_disconnect()
        self.health.on_reconnect_cycle_start()
        reconnect_attempts = int(self.pipeline_cfg.get("rtsp_reconnect_attempts", 30))
        reconnect_delay_sec = float(self.pipeline_cfg.get("rtsp_reconnect_delay_sec", 1.0))
        for attempt in range(reconnect_attempts):
            print(f"RTSP reconnect attempt {attempt + 1}/{reconnect_attempts}")
            self.health.on_reconnect_attempt()
            self.release()
            time.sleep(reconnect_delay_sec)
            if not self.open():
                continue
            success, frame = self.cap.read()
            if success:
                self.health.on_reconnect_success()
                self.health.on_frame_read_success()
                return True, frame
            self.health.on_frame_read_failure()
        self.health.on_reconnect_cycle_failed()
        return False, None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def draw_roi_layer(frame, roi_rect, roi_cfg: dict) -> None:
    if not roi_cfg["enabled"] or not roi_cfg.get("draw_enabled", False):
        return
    x1, y1, x2, y2 = roi_rect
    if roi_cfg.get("fill_alpha", 0.0) > 0.0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), roi_cfg["fill_color"], -1)
        cv2.addWeighted(overlay, roi_cfg["fill_alpha"], frame, 1.0 - roi_cfg["fill_alpha"], 0, frame)
    if roi_cfg.get("draw_border", True):
        cv2.rectangle(frame, (x1, y1), (x2, y2), roi_cfg["draw_color"], 2)
    if roi_cfg.get("draw_label", False):
        cv2.putText(
            frame,
            "ROI",
            (x1 + 6, min(y2 - 6, y1 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            roi_cfg["draw_color"],
            2,
        )


def _timestamp_from_capture(cap: cv2.VideoCapture, frame_idx: int, input_fps: float) -> float:
    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_ms > 0:
        return pos_ms / 1000.0
    fps = input_fps if input_fps > 0 else 30.0
    return frame_idx / float(fps)


def _write_event(csv_writer, jsonl_file, event: ViolationEvent, video_id: str) -> None:
    row = event.to_dict()
    enriched = {
        "video_id": video_id,
        "schema_version": PRED_EVENT_SCHEMA_VERSION,
        **row,
    }
    bbox = enriched.get("person_bbox")
    if bbox is None:
        bx1 = by1 = bx2 = by2 = ""
    else:
        bx1, by1, bx2, by2 = [f"{float(v):.2f}" for v in bbox]
    csv_writer.writerow(
        [
            enriched["video_id"],
            enriched["schema_version"],
            enriched["event_id"],
            enriched["frame_idx"],
            f"{float(enriched['timestamp_sec']):.3f}",
            enriched["person_track_id"],
            enriched["event_type"],
            enriched["no_hardhat_streak"],
            f"{float(enriched['no_hardhat_duration_sec']):.3f}",
            bx1, by1, bx2, by2,
        ]
    )
    jsonl_file.write(json.dumps(enriched, ensure_ascii=False, default=list) + "\n")


def _write_profile_report(report_path: str, report: dict) -> None:
    ensure_parent_dir(report_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _compute_auto_roi(source: VideoSource, cfg: dict, out_cfg: dict) -> tuple[dict, str | None]:
    roi_cfg = cfg["roi"]
    if not bool(roi_cfg.get("auto_enabled", False)):
        return {}, None
    cap = source.cap
    if cap is None:
        return {"enabled": False, "reason": "capture_not_open"}, None

    sample_frames = int(roi_cfg.get("auto_sample_frames", 48))
    min_motion_ratio = float(roi_cfg.get("auto_min_motion_ratio", 0.001))
    min_roi_ratio = float(roi_cfg.get("auto_min_roi_size_ratio", 0.50))
    margin_ratio = float(roi_cfg.get("auto_margin_ratio", 0.15))
    _, first = cap.read()
    if first is None:
        return {"enabled": False, "reason": "no_frames"}, None
    h, w = first.shape[:2]
    gray_prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    heat = np.zeros((h, w), dtype=np.float32)
    motion_pixels = 0
    frames_used = 0
    for _ in range(max(1, sample_frames - 1)):
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, gray_prev)
        _, m = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        motion_pixels += int(cv2.countNonZero(m))
        heat += (m > 0).astype(np.float32)
        gray_prev = gray
        frames_used += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if frames_used <= 0:
        return {"enabled": False, "reason": "insufficient_samples"}, None
    avg_motion_ratio = motion_pixels / float(frames_used * h * w)
    if avg_motion_ratio < min_motion_ratio:
        return {"enabled": False, "reason": "low_motion", "avg_motion_ratio": avg_motion_ratio}, None
    mask = heat >= max(1.0, frames_used * 0.08)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return {"enabled": False, "reason": "empty_motion_mask", "avg_motion_ratio": avg_motion_ratio}, None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w - 1, x2 + mx)
    y2 = min(h - 1, y2 + my)
    roi_area_ratio = ((x2 - x1 + 1) * (y2 - y1 + 1)) / float(w * h)
    if roi_area_ratio < min_roi_ratio:
        return {
            "enabled": False,
            "reason": "roi_too_small",
            "avg_motion_ratio": avg_motion_ratio,
            "roi_area_ratio": roi_area_ratio,
        }, None

    out_dir = Path(out_cfg["video_path"]).parent
    debug_img = np.zeros((h, w, 3), dtype=np.uint8)
    norm_heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    debug_img[:, :, 2] = norm_heat
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    debug_path = str((out_dir / "auto_roi_debug.jpg").resolve())
    cv2.imwrite(debug_path, debug_img)
    return {
        "enabled": True,
        "reason": "ok",
        "x_ratio": float(x1 / max(1, w)),
        "y_ratio": float(y1 / max(1, h)),
        "w_ratio": float((x2 - x1 + 1) / max(1, w)),
        "h_ratio": float((y2 - y1 + 1) / max(1, h)),
        "avg_motion_ratio": avg_motion_ratio,
        "roi_area_ratio": roi_area_ratio,
        "frames_used": frames_used,
    }, debug_path


def _binary_hardhat_observed_by_person(
    detections: list[Detection],
    person_boxes: dict[int, tuple[float, float, float, float]],
) -> dict[int, bool]:
    observed = {person_id: False for person_id in person_boxes}
    if not person_boxes:
        return observed
    for det in detections:
        if det.hardhat_state is None:
            continue
        owner_id = det.track_id if det.track_id in person_boxes else None
        if owner_id is None:
            x1, y1, x2, y2 = det.bbox_xyxy
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            for person_id, (px1, py1, px2, py2) in person_boxes.items():
                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    owner_id = person_id
                    break
        if owner_id is None:
            continue
        if det.hardhat_state == "with_hard_hat":
            observed[owner_id] = True
    return observed


def _vest_observed_by_person(
    detections: list[Detection],
    person_boxes: dict[int, tuple[float, float, float, float]],
) -> dict[int, bool]:
    observed = {person_id: False for person_id in person_boxes}
    if not person_boxes:
        return observed
    for det in detections:
        if det.cls_name != "vest":
            continue
        owner_id = det.owner_person_id if det.owner_person_id in person_boxes else None
        if owner_id is None:
            x1, y1, x2, y2 = det.bbox_xyxy
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            for person_id, (px1, py1, px2, py2) in person_boxes.items():
                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    owner_id = person_id
                    break
        if owner_id is None:
            continue
        observed[owner_id] = True
    return observed


class PipelineRunner:
    """Single end-to-end pipeline execution (video in → inference → events → artifacts out)."""

    __slots__ = ("cfg",)

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def run(self) -> dict:
        return _run_pipeline_inner(self.cfg)


def run_pipeline(cfg: dict) -> dict:
    return PipelineRunner(cfg).run()


def _run_pipeline_inner(cfg: dict) -> dict:
    pipeline_cfg = cfg["pipeline"]
    motion_cfg = cfg["motion"]
    roi_cfg = cfg["roi"]
    vis_cfg = cfg["visualization"]
    out_cfg = cfg["output"]
    profile_json_path = out_cfg.get("profile_json", "output_files/runtime_profile.json")
    metrics_flush_every = max(1, int(out_cfg.get("metrics_flush_every", 30)))

    ensure_parent_dir(out_cfg["video_path"])
    ensure_parent_dir(out_cfg["events_csv"])
    ensure_parent_dir(out_cfg["events_jsonl"])
    ensure_parent_dir(out_cfg["metrics_csv"])
    ensure_parent_dir(profile_json_path)

    source = VideoSource(pipeline_cfg["source"], pipeline_cfg)
    video_id = _resolve_video_id(
        source_value=pipeline_cfg["source"],
        explicit_video_id=str(pipeline_cfg.get("video_id", "") or "").strip(),
    )
    if not source.open():
        raise FileNotFoundError(f"Could not open input source: {pipeline_cfg['source']}")
    auto_roi_data, auto_roi_debug_img = _compute_auto_roi(source, cfg, out_cfg)
    if auto_roi_data.get("enabled", False):
        for key in ("x_ratio", "y_ratio", "w_ratio", "h_ratio"):
            roi_cfg[key] = float(auto_roi_data[key])
    out_dir_pre = Path(out_cfg["video_path"]).parent
    if auto_roi_debug_img is None:
        fallback_debug = np.zeros((int(pipeline_cfg["resize_to"][1]), int(pipeline_cfg["resize_to"][0]), 3), dtype=np.uint8)
        cv2.putText(
            fallback_debug,
            "AUTO ROI DISABLED",
            (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        auto_roi_debug_img = str((out_dir_pre / "auto_roi_debug.jpg").resolve())
        cv2.imwrite(auto_roi_debug_img, fallback_debug)

    resize_to = tuple(pipeline_cfg["resize_to"])
    writer, output_video_path, input_fps = create_video_writer(source.cap, out_cfg["video_path"], resize_to)

    detector = PPEDetector(cfg)
    tracker = PersonTracker(cfg, detector.class_ids)
    event_logic = TemporalEventLogic(cfg)
    profiler = RuntimeProfiler()
    motion_detector = MotionDetector(motion_cfg)
    sampler = FrameSampler(float(pipeline_cfg.get("sampling_fps", 0.0)))
    gate = InferenceGate(pipeline_cfg, motion_cfg)
    renderer = VisualizationRenderer(vis_cfg)

    frame_idx = 0
    total_events = 0
    last_infer_frame = -10**9
    max_frames = int(pipeline_cfg.get("max_frames", 0))

    last_main_detections: list[Detection] = []
    infer_frame_count = 0
    debug_filter_totals: dict[str, int] = {}
    infer_frames_with_zero_person = 0
    infer_frames_with_person = 0
    hardhat_detection_frames = 0
    hardhat_detection_total = 0
    head_detection_total = 0
    person_tracks_total_accum = 0
    unique_hardhat_person_ids: set[int] = set()
    unique_vest_person_ids: set[int] = set()
    skipped_stale_visual_bbox = 0
    person_confirm_mode = resolve_person_confirmation_mode(cfg["filters"])
    person_infer_hits: dict[int, int] = {}
    last_infer_person_confirm: dict[int, bool] = {}
    last_confirmed_person_ids: set[int] = set()

    with open(out_cfg["metrics_csv"], "w", newline="", encoding="utf-8") as metrics_f, open(
        out_cfg["events_csv"], "w", newline="", encoding="utf-8"
    ) as events_csv_f, open(out_cfg["events_jsonl"], "w", encoding="utf-8") as events_jsonl_f:
        metrics_writer = csv.writer(metrics_f)
        events_writer = csv.writer(events_csv_f)

        metrics_writer.writerow(metrics_csv_fieldnames())
        events_writer.writerow(
            [
                "video_id",
                "schema_version",
                "event_id",
                "frame_idx",
                "timestamp_sec",
                "person_track_id",
                "event_type",
                "no_hardhat_streak",
                "no_hardhat_duration_sec",
            ]
        )

        while True:
            loop_start = time.perf_counter()
            stop_requested = False
            io_ms = 0.0
            main_infer_ms = 0.0
            roi_infer_ms = 0.0
            tracking_ms = 0.0
            event_logic_ms = 0.0
            render_ms = 0.0

            t_io_read_start = time.perf_counter()
            success, frame = source.read()
            io_ms += (time.perf_counter() - t_io_read_start) * 1000.0
            if not success:
                break

            frame_idx += 1
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            timestamp_sec = _timestamp_from_capture(source.cap, frame_idx, input_fps)
            roi_rect = get_roi_rect(frame.shape[1], frame.shape[0], roi_cfg)

            motion_ratio = motion_detector.update(frame) if motion_cfg.get("enabled", True) else 0.0
            sampled = sampler.should_sample(timestamp_sec)
            should_infer = gate.should_infer(frame_idx, sampled, motion_ratio, last_infer_frame)
            infer_ms = 0.0

            person_boxes: dict[int, tuple[float, float, float, float]] = {}
            person_boxes_for_events: dict[int, tuple[float, float, float, float]] = {}
            confirmed_ids_for_draw: set[int] = set()
            person_hardhat_observed: dict[int, bool] = {}
            person_vest_observed: dict[int, bool] = {}
            accepted_heads = []
            accepted_hardhats = []
            roi_stage_enabled = bool(roi_cfg.get("enabled", True)) and bool(
                roi_cfg.get("person_roi_inference_enabled", True)
            )
            binary_mode = bool(getattr(detector, "binary_enabled", False))

            if should_infer:
                detections, main_infer_ms = detector.infer_main(frame, roi_rect)
                infer_ms += main_infer_ms
                last_main_detections = list(detections)
                infer_frame_count += 1

                t_track_update_start = time.perf_counter()
                person_boxes = tracker.update_person_tracks(detections, frame.shape, roi_rect, frame_idx)
                tracking_ms += (time.perf_counter() - t_track_update_start) * 1000.0
                person_stats = dict(getattr(tracker, "last_person_filter_stats", {}) or {})
                for key, val in person_stats.items():
                    debug_filter_totals[key] = int(debug_filter_totals.get(key, 0)) + int(val)
                person_tracks_total_accum += int(len(person_boxes))
                if len(person_boxes) == 0:
                    infer_frames_with_zero_person += 1
                else:
                    infer_frames_with_person += 1

                roi_detections = []
                headlike_detections = []
                if binary_mode:
                    person_hardhat_observed = _binary_hardhat_observed_by_person(detections, person_boxes)
                else:
                    if roi_stage_enabled:
                        t_roi_infer_start = time.perf_counter()
                        roi_detections = detector.infer_head_hardhat_in_person_rois(frame, person_boxes)
                        roi_infer_ms = (time.perf_counter() - t_roi_infer_start) * 1000.0
                        infer_ms += roi_infer_ms

                    headlike_detections = [d for d in detections if d.cls_name in {"head", "hardhat"}]
                    headlike_detections.extend(roi_detections)

                    t_track_assoc_start = time.perf_counter()
                    person_hardhat_observed, accepted_heads, accepted_hardhats = tracker.associate_head_hardhat(
                        headlike_detections,
                        person_boxes,
                        frame.shape,
                    )
                    hh_stats = dict(getattr(tracker, "last_head_hat_filter_stats", {}) or {})
                    for key, val in hh_stats.items():
                        debug_filter_totals[key] = int(debug_filter_totals.get(key, 0)) + int(val)
                    for pid, observed in person_hardhat_observed.items():
                        if observed:
                            unique_hardhat_person_ids.add(pid)
                    head_detection_total += int(len(accepted_heads))
                    hardhat_detection_total += int(len(accepted_hardhats))
                    if len(accepted_hardhats) > 0:
                        hardhat_detection_frames += 1
                    tracking_ms += (time.perf_counter() - t_track_assoc_start) * 1000.0
                person_vest_observed = _vest_observed_by_person(detections, person_boxes)
                for pid, observed in person_vest_observed.items():
                    if observed:
                        unique_vest_person_ids.add(pid)
                tracker.register_headlike_support(person_boxes, accepted_heads, accepted_hardhats)
                valid_person_ids = tracker.get_valid_person_ids(person_boxes)
                if valid_person_ids:
                    person_boxes = {pid: box for pid, box in person_boxes.items() if pid in valid_person_ids}
                    person_hardhat_observed = {
                        pid: person_hardhat_observed.get(pid, False) for pid in person_boxes
                    }
                    person_vest_observed = {
                        pid: person_vest_observed.get(pid, False) for pid in person_boxes
                    }
                last_infer_frame = frame_idx

                headlikes = [d for d in headlike_detections if d.cls_name in ("head", "hardhat")]
                filters_f = cfg["filters"]
                fh, fw = frame.shape[:2]
                n_raw_person = len(person_boxes)

                if person_confirm_mode == "off":
                    confirmed_ids_for_draw = set(person_boxes.keys())
                    person_boxes_for_events = dict(person_boxes)
                elif person_confirm_mode == "hard":
                    confirmed_map = {
                        pid: box
                        for pid, box in person_boxes.items()
                        if person_confirmed_by_head_or_hardhat(box, headlikes)
                    }
                    last_confirmed_person_ids = set(confirmed_map.keys())
                    n_conf_log = len(confirmed_map)
                    n_weak_log = 0
                    n_rej_log = n_raw_person - n_conf_log
                    person_boxes = confirmed_map
                    person_hardhat_observed = {
                        pid: person_hardhat_observed[pid] for pid in person_boxes if pid in person_hardhat_observed
                    }
                    accepted_heads = [h for h in accepted_heads if h.owner_person_id in person_boxes]
                    accepted_hardhats = [h for h in accepted_hardhats if h.owner_person_id in person_boxes]
                    confirmed_ids_for_draw = set(person_boxes.keys())
                    person_boxes_for_events = dict(person_boxes)
                    if pipeline_cfg.get("debug_visualization", False):
                        print(
                            f"[person-confirm] raw_person_count={n_raw_person} "
                            f"confirmed_person_count={n_conf_log} weak_person_count={n_weak_log} "
                            f"rejected_person_count={n_rej_log}",
                            flush=True,
                        )
                else:  # soft
                    for pid in person_boxes:
                        person_infer_hits[pid] = person_infer_hits.get(pid, 0) + 1
                    hi_th = float(filters_f["person_high_conf_threshold"])
                    min_asp = float(filters_f["person_min_aspect_ratio"])
                    min_ar = float(filters_f["person"]["min_area_ratio"])
                    min_hits = int(filters_f["person_min_confirmed_track_hits"])
                    last_infer_person_confirm = {}
                    confirmed_ids_for_draw = set()
                    for pid, box in person_boxes.items():
                        pconf = float(tracker.last_person_conf.get(pid, 0.0))
                        hits = person_infer_hits[pid]
                        ok = person_soft_confirmed(
                            box,
                            headlikes,
                            pconf,
                            fw,
                            fh,
                            hits,
                            high_conf_threshold=hi_th,
                            min_aspect_hw=min_asp,
                            min_area_ratio=min_ar,
                            min_infer_hits=min_hits,
                        )
                        last_infer_person_confirm[pid] = ok
                        if ok:
                            confirmed_ids_for_draw.add(pid)
                    n_conf_log = len(confirmed_ids_for_draw)
                    n_weak_log = n_raw_person - n_conf_log
                    n_rej_log = 0
                    person_boxes_for_events = {pid: person_boxes[pid] for pid in confirmed_ids_for_draw}
                    if pipeline_cfg.get("debug_visualization", False):
                        print(
                            f"[person-confirm] raw_person_count={n_raw_person} "
                            f"confirmed_person_count={n_conf_log} weak_person_count={n_weak_log} "
                            f"rejected_person_count={n_rej_log}",
                            flush=True,
                        )
            else:
                t_track_reuse_start = time.perf_counter()
                raw_person_boxes = tracker.get_active_person_boxes(
                    frame_idx,
                    int(pipeline_cfg.get("reuse_tracks_max_gap_frames", 8)),
                )
                person_boxes = dict(raw_person_boxes)
                person_visual_ttl = int(vis_cfg.get("person_visual_ttl_frames", 1))
                if person_visual_ttl >= 0:
                    person_boxes = {
                        pid: box
                        for pid, box in person_boxes.items()
                        if (frame_idx - tracker.last_seen_frame.get(pid, -10**9)) <= person_visual_ttl
                    }
                    skipped_stale_visual_bbox += max(0, len(raw_person_boxes) - len(person_boxes))
                valid_person_ids = tracker.get_valid_person_ids(person_boxes)
                if valid_person_ids:
                    person_boxes = {pid: box for pid, box in person_boxes.items() if pid in valid_person_ids}
                else:
                    person_boxes = {}
                person_hardhat_observed = {person_id: False for person_id in person_boxes}
                person_vest_observed = {person_id: False for person_id in person_boxes}
                tracking_ms += (time.perf_counter() - t_track_reuse_start) * 1000.0
                if person_confirm_mode == "hard":
                    person_boxes = {
                        pid: box
                        for pid, box in person_boxes.items()
                        if pid in last_confirmed_person_ids
                    }
                    person_hardhat_observed = {person_id: False for person_id in person_boxes}
                    person_vest_observed = {person_id: False for person_id in person_boxes}
                    confirmed_ids_for_draw = set(person_boxes.keys())
                    person_boxes_for_events = dict(person_boxes)
                elif person_confirm_mode == "soft":
                    confirmed_ids_for_draw = {
                        pid for pid in person_boxes if last_infer_person_confirm.get(pid, False)
                    }
                    person_boxes_for_events = {pid: person_boxes[pid] for pid in confirmed_ids_for_draw}
                else:
                    confirmed_ids_for_draw = set(person_boxes.keys())
                    person_boxes_for_events = dict(person_boxes)

            person_hardhat_for_events = {
                pid: person_hardhat_observed[pid]
                for pid in person_boxes_for_events
                if pid in person_hardhat_observed
            }
            person_vest_for_events = {
                pid: person_vest_observed[pid] for pid in person_boxes_for_events if pid in person_vest_observed
            }

            t_event_logic_start = time.perf_counter()
            statuses, events, active_violations, violating_person_ids = event_logic.update(
                person_boxes=person_boxes_for_events,
                person_hardhat_observed=person_hardhat_for_events,
                person_vest_observed=person_vest_for_events,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                did_infer=should_infer,
            )
            event_logic_ms = (time.perf_counter() - t_event_logic_start) * 1000.0

            if pipeline_cfg.get("debug_visualization", False):
                _dbe = max(1, int(pipeline_cfg.get("debug_log_every", 30)))
                if frame_idx % _dbe == 0 or should_infer:
                    pstats = dict(getattr(tracker, "last_person_filter_stats", {}) or {})
                    print(
                        f"[viz-debug] frame={frame_idx} infer={int(should_infer)} "
                        f"person_boxes={len(person_boxes)} "
                        f"acc_heads={len(accepted_heads)} acc_hardhats={len(accepted_hardhats)} "
                        f"raw_main={len(last_main_detections)} person_filter_stats={pstats}",
                        flush=True,
                    )

            t_render_start = time.perf_counter()
            renderer.draw_runtime_overlay(
                frame=frame,
                last_main_detections=last_main_detections,
                accepted_heads=accepted_heads,
                accepted_hardhats=accepted_hardhats,
                person_boxes=person_boxes,
                confirmed_ids_for_draw=confirmed_ids_for_draw,
                statuses=statuses,
                event_logic=event_logic,
                violating_person_ids=violating_person_ids,
                frame_idx=frame_idx,
                input_fps=input_fps,
                person_confirm_mode=person_confirm_mode,
                active_violations=active_violations,
                total_events=total_events + len(events),
                person_hardhat_observed=person_hardhat_observed,
                active_no_hardhat_now=len(event_logic.active_no_hardhat_persons),
                unique_no_hardhat_total=len(event_logic.unique_no_hardhat_persons),
            )

            draw_roi_layer(frame, roi_rect, roi_cfg)
            render_ms = (time.perf_counter() - t_render_start) * 1000.0

            t_io_out_start = time.perf_counter()
            for event in events:
                _write_event(events_writer, events_jsonl_f, event, video_id=video_id)
            if events:
                events_csv_f.flush()
                events_jsonl_f.flush()
            total_events += len(events)

            removed_ids = tracker.prune_stale_tracks(
                frame_idx=frame_idx,
                max_stale_frames=int(pipeline_cfg.get("max_track_stale_frames", 45)),
            )
            if removed_ids:
                event_logic.remove_ids(removed_ids)
                for rid in removed_ids:
                    person_infer_hits.pop(rid, None)
                    last_infer_person_confirm.pop(rid, None)

            writer.write(frame)
            if pipeline_cfg.get("display_preview", True):
                cv2.imshow("PPE Monitoring Runtime", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_requested = True
            io_ms += (time.perf_counter() - t_io_out_start) * 1000.0

            total_loop_ms = (time.perf_counter() - loop_start) * 1000.0
            loop_ms = total_loop_ms
            stage_timings = {
                "main_infer_ms": main_infer_ms,
                "roi_infer_ms": roi_infer_ms,
                "tracking_ms": tracking_ms,
                "event_logic_ms": event_logic_ms,
                "render_ms": render_ms,
                "io_ms": io_ms,
                "total_loop_ms": total_loop_ms,
            }
            profiler.record_frame(
                loop_ms=loop_ms,
                did_infer=should_infer,
                infer_ms=infer_ms,
                motion_ratio=motion_ratio,
                stage_timings_ms=stage_timings,
            )

            metrics_writer.writerow(
                [
                    frame_idx,
                    f"{timestamp_sec:.3f}",
                    f"{motion_ratio:.6f}",
                    int(sampled),
                    int(should_infer),
                    f"{infer_ms:.3f}",
                    f"{loop_ms:.3f}",
                    f"{main_infer_ms:.3f}",
                    f"{roi_infer_ms:.3f}",
                    f"{tracking_ms:.3f}",
                    f"{event_logic_ms:.3f}",
                    f"{render_ms:.3f}",
                    f"{io_ms:.3f}",
                    f"{total_loop_ms:.3f}",
                    len(person_boxes),
                    active_violations,
                    len(events),
                    total_events,
                ]
            )
            if frame_idx % metrics_flush_every == 0:
                metrics_f.flush()

            if stop_requested:
                break
            if max_frames > 0 and frame_idx >= max_frames:
                break

        if pipeline_cfg.get("debug_visualization", False) and frame_idx > 0:
            print(
                f"[viz-debug] summary: frames={frame_idx} infer_frames={infer_frame_count} "
                f"infer_frame_ratio={infer_frame_count / float(frame_idx):.4f}",
                flush=True,
            )

        metrics_f.flush()

    summary = profiler.summary(input_fps=input_fps)
    source.health.finalize()
    rtsp_health = source.health.to_dict()
    fps_gap = float(summary["input_fps"]) - float(summary["inference_fps"])
    inference_to_input_ratio = (
        float(summary["inference_fps"]) / float(summary["input_fps"])
        if float(summary["input_fps"]) > 0
        else 0.0
    )
    profile_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(pipeline_cfg["source"]),
        "is_rtsp": rtsp_health["is_rtsp"],
        "events_total": total_events,
        "runtime_summary": summary,
        "comparison": {
            "input_fps_minus_inference_fps": fps_gap,
            "inference_fps_to_input_fps_ratio": inference_to_input_ratio,
        },
        "rtsp_health": rtsp_health,
        "output_paths": {
            "configured_video_path": str(out_cfg["video_path"]),
            "actual_video_path": str(output_video_path),
        },
        "metrics_csv_columns": metrics_csv_fieldnames(),
        "stage_timing_field_order": list(STAGE_TIMING_FIELDS),
        "detection_debug": {
            "frames_total": int(frame_idx),
            "infer_frames_total": int(infer_frame_count),
            "infer_frames_with_person": int(infer_frames_with_person),
            "infer_frames_with_zero_person": int(infer_frames_with_zero_person),
            "avg_person_tracks_per_infer_frame": (
                float(person_tracks_total_accum) / float(max(1, infer_frame_count))
            ),
            "head_detection_total": int(head_detection_total),
            "hardhat_detection_frames": int(hardhat_detection_frames),
            "hardhat_detection_total": int(hardhat_detection_total),
            "unique_with_hardhat_persons_total": int(len(unique_hardhat_person_ids)),
            "unique_with_vest_persons_total": int(len(unique_vest_person_ids)),
            "active_no_hardhat_persons_now": int(len(event_logic.active_no_hardhat_persons)),
            "unique_no_hardhat_persons_total": int(len(event_logic.unique_no_hardhat_persons)),
            "active_no_vest_persons_now": int(len(event_logic.active_no_vest_persons)),
            "unique_no_vest_persons_total": int(len(event_logic.unique_no_vest_persons)),
            "skipped_stale_visual_bbox": int(skipped_stale_visual_bbox),
            "person_filter_totals": debug_filter_totals,
        },
        "auto_roi": auto_roi_data,
        "config": cfg,
    }
    _write_profile_report(profile_json_path, profile_report)
    out_dir = Path(out_cfg["video_path"]).parent
    auto_roi_json = str((out_dir / "auto_roi.json").resolve())
    with open(auto_roi_json, "w", encoding="utf-8") as f:
        json.dump(auto_roi_data, f, ensure_ascii=False, indent=2)
    detection_diag_json = str((out_dir / "detection_diagnostics.json").resolve())
    with open(detection_diag_json, "w", encoding="utf-8") as f:
        json.dump(profile_report["detection_debug"], f, ensure_ascii=False, indent=2)

    source.release()
    writer.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        # Some headless OpenCV builds do not implement GUI teardown calls.
        pass

    print(f"Input source: {pipeline_cfg['source']}")
    print(f"Output video: {output_video_path}")
    print(f"Input FPS: {summary['input_fps']:.2f}")
    print(f"Processing FPS: {summary['processing_fps']:.2f}")
    print(f"Inference FPS: {summary['inference_fps']:.2f}")
    print(
        "Inference latency ms (mean/median/p90): "
        f"{summary['inference_ms_mean']:.2f}/{summary['inference_ms_median']:.2f}/{summary['inference_ms_p90']:.2f}"
    )
    print(
        "Loop latency ms (mean/median/p90): "
        f"{summary['loop_ms_mean']:.2f}/{summary['loop_ms_median']:.2f}/{summary['loop_ms_p90']:.2f}"
    )
    print(f"Total events: {total_events}")
    print(f"Events CSV: {out_cfg['events_csv']}")
    print(f"Events JSONL: {out_cfg['events_jsonl']}")
    print(f"Metrics CSV: {out_cfg['metrics_csv']}")
    print(f"Profile JSON: {profile_json_path}")
    if rtsp_health["is_rtsp"]:
        print(
            "RTSP health reconnects (attempts/success/fail): "
            f"{rtsp_health['reconnect_attempts_total']}/"
            f"{rtsp_health['reconnect_successes']}/"
            f"{rtsp_health['reconnect_failures']}"
        )

    return {
        "summary": summary,
        "events_total": total_events,
        "rtsp_health": rtsp_health,
        "output_video_path": output_video_path,
        "events_csv": out_cfg["events_csv"],
        "events_jsonl": out_cfg["events_jsonl"],
        "metrics_csv": out_cfg["metrics_csv"],
        "profile_json": profile_json_path,
        "detection_diagnostics_json": detection_diag_json,
        "auto_roi_json": auto_roi_json,
        "auto_roi_debug_image": auto_roi_debug_img,
        "detection_debug": profile_report["detection_debug"],
        "auto_roi": auto_roi_data,
    }
