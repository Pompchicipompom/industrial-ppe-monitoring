from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "pipeline": {
        "source": "input_files/hardhat_input_video4.mp4",  # file path, camera index, or rtsp://...
        "resize_to": (640, 640),
        "display_preview": False,
        "sampling_fps": 15.0,  # Frame sampling target. <=0 means process every input frame.
        "mode": "motion_gated",  # "every_sample" | "motion_gated"
        "force_infer_every_n_frames": 12,
        # Keep last boxes on non-infer frames; must be >= typical gap between inferences (motion_gated + force_infer).
        "reuse_tracks_max_gap_frames": 45,
        "max_track_stale_frames": 45,
        "max_frames": 0,  # 0 means process full stream.
        "rtsp_reconnect_attempts": 30,
        "rtsp_reconnect_delay_sec": 1.0,
        # stderr: per-frame viz stats when True (see debug_log_every).
        "debug_visualization": False,
        "debug_log_every": 30,
        "debug_person_filter_stats": True,
    },
    "motion": {
        "enabled": True,  # Fixed camera assumption: infer mostly on motion.
        "pixel_threshold": 20,
        "min_ratio": 0.0007,
        "min_ratio_off": 0.0004,
        "hold_frames_after_motion": 15,
        "blur_kernel": 5,
        "background_alpha": 0.03,
        "use_morphology": True,
        "morph_kernel": 3,
    },
    "model": {
        "detector_backend": "triple_class",  # "triple_class" | "binary_hardhat"
        "weights_path": "models/hardhat_detection_yolo11_200_epochs_best_02032025.pt",
        "weights_path_lite": "",
        "prefer_lite_model": False,
        "auto_backend_resolve": True,
        "backend_priority": ["engine", "openvino", "onnx", "pt"],
        "conf_threshold": 0.18,
        "person_min_conf": 0.22,
        "roi_head_hardhat_conf": 0.10,
        "imgsz": 640,
        "roi_imgsz": 320,
        "max_det": 300,
        "device": "",  # "", "cpu", "0"
        "use_half": True,
        "enable_person_fallback": True,
        "person_fallback_weights_path": "yolov8s.pt",
        "person_fallback_conf": 0.12,
        "person_fallback_imgsz": 640,
        # Binary hardhat backend (e.g., with_hard_hat / without_hard_hat)
        "binary_weights_path": "models/hardhat_binary_best.pt",
        "binary_conf_threshold": 0.35,
        "binary_imgsz": 512,
        "binary_class_with_hardhat": "with_hard_hat",
        "binary_class_without_hardhat": "without_hard_hat",
        # Optional class-name aliases for dataset compatibility (e.g. Construction-PPE).
        # Mapping form: model_label -> canonical_label ("person"|"head"|"hardhat"|"vest").
        "class_name_aliases": {
            "Person": "person",
            "helmet": "hardhat",
        },
        # stderr: log class counts after each main-frame infer (verbose).
        "debug_log_infer": False,
    },
    "roi": {
        "enabled": True,
        "auto_enabled": False,
        "auto_sample_frames": 48,
        "auto_min_motion_ratio": 0.001,
        "auto_min_roi_size_ratio": 0.50,
        "auto_margin_ratio": 0.15,
        "person_roi_inference_enabled": True,
        "global_inference_in_roi": True,
        "person_center_must_be_in_roi": True,
        "reject_person_boxes_on_roi_edge": False,
        "person_edge_reject_margin_px": 8,
        "x_ratio": 0.00,
        "y_ratio": 0.50,
        "w_ratio": 1.00,
        "h_ratio": 0.50,
        "draw_enabled": False,
        "draw_border": True,
        "draw_label": False,
        "draw_color": (0, 255, 255),
        "fill_color": (0, 255, 255),
        "fill_alpha": 0.0,
    },
    "filters": {
        # Person gating for demo/debug: off | hard | soft (E2 default off).
        # Legacy: require_head_or_hardhat_for_person=true implies hard when mode is off.
        "person_confirmation_mode": "off",
        "require_head_or_hardhat_for_person": False,
        "person_high_conf_threshold": 0.55,
        # height/width; tall bbox (person-like silhouette).
        "person_min_aspect_ratio": 1.4,
        "person_min_confirmed_track_hits": 3,
        "person": {
            "min_area_ratio": 0.001,
            "max_area_ratio": 0.22,
            "min_aspect_ratio": 0.15,
            "max_aspect_ratio": 0.90,
            "max_width_ratio": 0.55,
            "min_height_ratio": 0.06,
            "max_height_ratio": 0.90,
        },
        "head_hardhat": {
            "min_area_ratio": 0.00003,
            "max_area_ratio": 0.030,
            "min_aspect_ratio": 0.15,
            "max_aspect_ratio": 2.60,
            "max_width_ratio": 0.22,
            "max_height_ratio": 0.30,
        },
        "head_vs_person": {
            "max_area_ratio_of_person": 0.45,
            "max_width_ratio_of_person": 0.95,
            "max_height_ratio_of_person": 0.70,
        },
    },
    "person_roi": {
        "x_expand_ratio": 0.16,
        "y_expand_top_ratio": 0.10,
        "y_bottom_ratio": 0.60,
    },
    "tracking": {
        "history_len": 10,
        "max_area_growth_ratio": 1.9,
        "max_width_growth_ratio": 1.45,
        "max_height_growth_ratio": 1.45,
        "max_person_area_ratio": 0.55,
        "max_area_jump_ratio": 2.8,
        "max_width_jump_ratio": 2.5,
        "max_height_jump_ratio": 2.5,
        "huge_bbox_confirm_frames": 3,
        "min_iou_if_growth_triggered": 0.12,
        "blend_alpha": 0.35,
        "max_center_shift_ratio_w": 0.65,
        "max_center_shift_ratio_h": 0.50,
        "transfer_iou_threshold": 0.35,
        "transfer_max_center_shift_ratio_w": 0.55,
        "transfer_max_center_shift_ratio_h": 0.45,
        "transfer_match_expand_ratio": 0.25,
        "transfer_require_area_consistency": True,
        "transfer_single_person_center_scale": 1.35,
        "transfer_min_iou_for_center_fallback": 0.0,
        "transfer_max_area_ratio_jump": 2.2,
        "max_transfer_gap_frames": 20,
        "headlike_dedup_iou": 0.45,
        "min_track_hits_for_valid_person": 3,
        "max_no_headlike_streak_frames": 8,
        "valid_person_high_conf_override": 0.70,
    },
    "event_logic": {
        "hardhat_confirm_frames": 2,
        "hardhat_revoke_frames": 4,
        "lock_after_confirm": True,
        "no_hardhat_consecutive_frames": 20,  # Temporal smoothing K.
        "no_hardhat_seconds_threshold": 2.0,  # Time threshold for industrial semantics.
        "no_vest_consecutive_frames": 12,
        "no_vest_seconds_threshold": 1.5,
        "cooldown_frames": 90,
        "cooldown_seconds": 3.0,
        "max_no_hardhat_hold_frames_without_infer": 4,
        "max_no_vest_hold_frames_without_infer": 4,
    },
    "visualization": {
        # Main person boxes from tracker (thick).
        "draw_tracker_boxes": True,
        "draw_person_boxes": True,
        # Accepted head/hardhat boxes after association/filtering (thin).
        "draw_head_hardhat_boxes": True,
        # Debug-only raw detections overlay from main infer.
        # Note: raw person/head/hardhat are intentionally suppressed in renderer to avoid duplicates.
        "draw_raw_detections": False,
        # Legacy alias kept for backward compatibility.
        "draw_all_main_detections": False,
        "draw_detection_confidence_labels": False,
        # soft mode: draw thin dashed box for unconfirmed tracks (demo only).
        "show_weak_person": False,
        "show_violation_banner": True,
        "show_violation_dot": True,
        "violation_text": "Нарушение: нет каски",
        "panel_attention_text": "Внимание",
        "panel_attention_sticky": True,
        "panel_blink_period_frames": 14,
        "dot_radius": 8,
        "dot_blink_period_frames": 14,
        "bbox_thickness": 2,
        "person_color": (255, 0, 0),
        "head_color": (0, 255, 255),
        "hardhat_color": (0, 255, 0),
        "vest_color": (255, 255, 0),
        "no_vest_color": (0, 165, 255),
        "violation_color": (0, 0, 255),
        "text_color": (255, 255, 255),
        "hide_panel_when_idle": False,
        "smoothing_enabled": True,
        "bbox_smoothing_alpha": 0.4,
        "bbox_ttl_frames": 3,
        "head_hat_ttl_frames": 4,
        "person_ttl_frames": 12,
        "person_visual_ttl_frames": 1,
        "bbox_smoothing_match_iou": 0.20,
        # Keep violation color when Ultralytics changes track_id but bbox overlaps recent violator.
        "violation_spatial_inherit_enabled": True,
        "violation_spatial_anchor_ttl_frames": 150,
        "violation_spatial_iou_min": 0.12,
        "violation_spatial_expand_ratio": 0.35,
        "violation_spatial_max_anchors": 50,
    },
    "output": {
        "video_path": "output_files/processed.mp4",
        "events_csv": "output_files/events.csv",
        "events_jsonl": "output_files/events.jsonl",
        "metrics_csv": "output_files/frame_metrics.csv",
        "profile_json": "output_files/runtime_profile.json",
        "metrics_flush_every": 30,  # flush frame_metrics.csv every N frames (1 = every frame)
    },
}


REQUIRED_CONFIG_TOP_KEYS: frozenset[str] = frozenset(DEFAULT_CONFIG.keys())


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate merged config: required sections, unknown top-level keys, basic types."""
    if not isinstance(cfg, dict):
        raise TypeError("config must be a dict")
    missing = sorted(REQUIRED_CONFIG_TOP_KEYS - set(cfg.keys()))
    if missing:
        raise ValueError(f"Config missing required top-level keys: {missing}")
    extra = sorted(set(cfg.keys()) - REQUIRED_CONFIG_TOP_KEYS)
    if extra:
        raise ValueError(
            f"Unknown top-level config keys (possible typo): {extra}. "
            f"Allowed keys: {sorted(REQUIRED_CONFIG_TOP_KEYS)}"
        )
    for section in REQUIRED_CONFIG_TOP_KEYS:
        if not isinstance(cfg[section], dict):
            raise TypeError(f"config[{section!r}] must be a dict, got {type(cfg[section]).__name__}")
    pl = cfg["pipeline"]
    for key in ("source", "mode"):
        if key not in pl or pl[key] is None or str(pl[key]).strip() == "":
            raise ValueError(f"pipeline.{key} is required")
    sf = pl.get("sampling_fps", 0.0)
    if not isinstance(sf, (int, float)):
        raise TypeError(f"pipeline.sampling_fps must be numeric, got {type(sf).__name__}")
    mf = pl.get("max_frames", 0)
    if not isinstance(mf, int):
        try:
            int(mf)
        except Exception as exc:
            raise TypeError("pipeline.max_frames must be int-compatible") from exc
    model_cfg = cfg["model"]
    detector_backend = str(model_cfg.get("detector_backend", "triple_class")).strip().lower()
    if detector_backend not in {"triple_class", "binary_hardhat"}:
        raise ValueError(
            "model.detector_backend must be one of: 'triple_class', 'binary_hardhat'"
        )


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        suffix = config_file.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except Exception as exc:
                raise RuntimeError(
                    "PyYAML is required to load YAML configs. Install with: pip install pyyaml"
                ) from exc
            user_cfg = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        elif suffix == ".json":
            import json

            user_cfg = json.loads(config_file.read_text(encoding="utf-8"))
        else:
            raise ValueError("Only .yaml/.yml/.json configs are supported.")
        _deep_update(cfg, user_cfg)
    if overrides:
        _deep_update(cfg, overrides)
    # Backward-compatible alias: allow model.model_path in external YAML.
    model_cfg = cfg.get("model", {})
    model_path = str(model_cfg.get("model_path", "") or "").strip()
    if model_path:
        model_cfg["weights_path"] = model_path
    validate_config(cfg)
    return cfg
