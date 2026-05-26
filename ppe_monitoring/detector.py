from __future__ import annotations

"""Ultralytics YOLO wrappers for PPE (person / head / hardhat).

Why **three** ``YOLO`` instances when ``enable_person_fallback`` is on:

1. **``self.model``** — full-frame ``track()`` with persistent tracker state for the custom PPE weights.
2. **``self.roi_model``** — same weights file but a **separate** object used only for ``predict()`` on person
   ROI crops (head/hardhat). This avoids cross-talk with internal tracker buffers tied to the main ``track()`` pass.
3. **``self.fallback_model``** — optional COCO ``person`` detector (e.g. ``yolov8s.pt``) merged with dedup rules.

Trade-off: higher RAM and startup cost vs. a single shared instance; prioritises correctness of tracking vs. ROI passes.
"""

import time
from pathlib import Path

from ultralytics import YOLO

from .geometry import bbox_iou, clip_xyxy_to_frame
from .types import Detection

try:
    import torch
except Exception:
    torch = None


class PPEDetector:
    DEFAULT_CLASS_NAME_ALIASES = {
        "person": "person",
        "Person": "person",
        "head": "head",
        "hardhat": "hardhat",
        "helmet": "hardhat",
        "vest": "vest",
        "safety vest": "vest",
        "safety_vest": "vest",
        "no-safety vest": "no_vest",
        "no_safety_vest": "no_vest",
        "no-vest": "no_vest",
        "no_vest": "no_vest",
    }

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_cfg = cfg["model"]
        self.roi_cfg = cfg["roi"]
        self.person_roi_cfg = cfg["person_roi"]
        self.detector_backend = str(self.model_cfg.get("detector_backend", "triple_class")).strip().lower()
        self.binary_enabled = self.detector_backend == "binary_hardhat"
        self.infer_imgsz = int(self.model_cfg.get("imgsz", 640))
        self.roi_imgsz = int(self.model_cfg.get("roi_imgsz", 320))
        self.fallback_imgsz = int(self.model_cfg.get("person_fallback_imgsz", self.infer_imgsz))
        self.max_det = int(self.model_cfg.get("max_det", 300))
        self.binary_conf_threshold = float(self.model_cfg.get("binary_conf_threshold", 0.35))
        self.binary_imgsz = int(self.model_cfg.get("binary_imgsz", self.infer_imgsz))
        self.device = str(self.model_cfg.get("device", "")).strip()
        self.use_half = bool(self.model_cfg.get("use_half", False))
        if self.use_half and (torch is None or not torch.cuda.is_available()):
            self.use_half = False

        self.weights_path = (
            str(self.model_cfg.get("binary_weights_path", "")).strip()
            if self.binary_enabled
            else self._resolve_weights_path()
        )
        if self.binary_enabled:
            if not self.weights_path:
                raise ValueError("Binary backend enabled but model.binary_weights_path is empty.")
            if not Path(self.weights_path).exists():
                print(
                    "WARNING: binary backend requested but weights not found; "
                    "falling back to triple_class backend."
                )
                self.binary_enabled = False
                self.detector_backend = "triple_class"
                self.weights_path = self._resolve_weights_path()
        self.artifact_kind = self._artifact_kind(self.weights_path)
        if self.artifact_kind in {"openvino", "onnx", "engine"} and self.roi_imgsz != self.infer_imgsz:
            # Exported accelerated artifacts are often static-shape; keep ROI inference shape compatible.
            self.roi_imgsz = self.infer_imgsz
        print(f"Detector artifact: {self.weights_path}")
        self.model = YOLO(self.weights_path, task="detect")
        # Separate model instance for person ROI inference to avoid tracker-state conflicts.
        self.roi_model = None if self.binary_enabled else YOLO(self.weights_path, task="detect")

        self.fallback_model = None
        if (not self.binary_enabled) and self.model_cfg.get("enable_person_fallback", True):
            self.fallback_model = YOLO(self.model_cfg["person_fallback_weights_path"], task="detect")

        self.class_name_aliases = self._build_class_aliases(
            self.model_cfg.get("class_name_aliases", {})
        )
        self.class_ids = self._get_class_ids(
            self.model.names,
            binary_enabled=self.binary_enabled,
            class_name_aliases=self.class_name_aliases,
        )
        self.id_to_name = self._build_id_to_name(self.model.names)
        self.binary_with_name = str(self.model_cfg.get("binary_class_with_hardhat", "with_hard_hat")).strip()
        self.binary_without_name = str(self.model_cfg.get("binary_class_without_hardhat", "without_hard_hat")).strip()
        self.binary_with_aliases = {
            self.binary_with_name.strip().lower().replace("-", "_"),
            "hardhat",
            "with_hard_hat",
        }
        self.binary_without_aliases = {
            self.binary_without_name.strip().lower().replace("-", "_"),
            "without_hard_hat",
            "no_hardhat",
            "nohelmet",
            "no_helmet",
        }

    @staticmethod
    def _artifact_kind(weights_path: str) -> str:
        p = Path(weights_path)
        suffix = p.suffix.lower()
        if suffix == ".engine":
            return "engine"
        if suffix == ".onnx":
            return "onnx"
        if suffix == ".pt":
            return "pt"
        if p.is_dir() and p.name.endswith("_openvino_model"):
            return "openvino"
        return "other"

    def _resolve_weights_path(self) -> str:
        prefer_lite = bool(self.model_cfg.get("prefer_lite_model", False))
        lite_path = str(self.model_cfg.get("weights_path_lite", "")).strip()
        base_path = lite_path if (prefer_lite and lite_path) else str(self.model_cfg["weights_path"])
        path = Path(base_path)

        if not bool(self.model_cfg.get("auto_backend_resolve", True)):
            return str(path)

        # If user already points to concrete artifact, keep it.
        if path.suffix.lower() in {".onnx", ".engine", ".xml"} or path.is_dir():
            return str(path)

        if path.suffix.lower() != ".pt":
            return str(path)

        stem = path.with_suffix("")
        backend_priority = list(self.model_cfg.get("backend_priority", ["engine", "openvino", "onnx", "pt"]))
        candidates: list[Path] = []
        for backend in backend_priority:
            key = str(backend).lower()
            if key == "engine":
                candidates.append(path.with_suffix(".engine"))
            elif key == "openvino":
                candidates.append(Path(f"{stem}_openvino_model"))
            elif key == "onnx":
                candidates.append(path.with_suffix(".onnx"))
            elif key == "pt":
                candidates.append(path)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return str(path)

    def _runtime_kwargs(self) -> dict:
        kwargs = {
            "imgsz": self.infer_imgsz,
            "max_det": self.max_det,
        }
        if self.device:
            kwargs["device"] = self.device
        if self.use_half:
            kwargs["half"] = True
        return kwargs

    @staticmethod
    def _build_id_to_name(model_names):
        if isinstance(model_names, dict):
            return {int(k): str(v) for k, v in model_names.items()}
        return {idx: str(name) for idx, name in enumerate(model_names)}

    @classmethod
    def _build_class_aliases(cls, config_aliases) -> dict[str, str]:
        aliases = dict(cls.DEFAULT_CLASS_NAME_ALIASES)
        if isinstance(config_aliases, dict):
            for raw_name, canonical in config_aliases.items():
                aliases[str(raw_name)] = str(canonical).strip().lower()
        return aliases

    def _normalize_class_name(self, model_cls_name: str) -> str:
        if model_cls_name in self.class_name_aliases:
            return self.class_name_aliases[model_cls_name]
        lowered = str(model_cls_name).strip().lower()
        if lowered in self.class_name_aliases:
            return self.class_name_aliases[lowered]
        return lowered

    @staticmethod
    def _get_class_ids(model_names, binary_enabled: bool = False, class_name_aliases: dict[str, str] | None = None):
        if isinstance(model_names, dict):
            raw_name_to_id = {str(v): int(k) for k, v in model_names.items()}
        else:
            raw_name_to_id = {str(name): idx for idx, name in enumerate(model_names)}
        aliases = class_name_aliases or {}
        name_to_id: dict[str, int] = {}
        for raw_name, cls_id in raw_name_to_id.items():
            canonical = aliases.get(raw_name)
            if canonical is None:
                canonical = aliases.get(str(raw_name).strip().lower(), str(raw_name).strip().lower())
            name_to_id.setdefault(str(canonical), int(cls_id))
        if binary_enabled:
            return {"person": 0, "head": -1, "hardhat": -1, "vest": -1, "no_vest": -1}
        required = ("person", "hardhat")
        missing = [name for name in required if name not in name_to_id]
        if missing:
            raise ValueError(f"Model classes missing required labels: {missing}")
        class_ids = {name: name_to_id[name] for name in required}
        class_ids["head"] = int(name_to_id.get("head", -1))
        class_ids["vest"] = int(name_to_id.get("vest", -1))
        class_ids["no_vest"] = int(name_to_id.get("no_vest", -1))
        return class_ids

    def infer_main(
        self,
        frame,
        roi_rect: tuple[int, int, int, int],
    ) -> tuple[list[Detection], float]:
        if self.binary_enabled:
            return self._infer_main_binary(frame=frame, roi_rect=roi_rect)

        infer_x1, infer_y1 = 0, 0
        infer_frame = frame
        if self.roi_cfg["enabled"] and self.roi_cfg.get("global_inference_in_roi", False):
            rx1, ry1, rx2, ry2 = roi_rect
            if (rx2 - rx1) >= 8 and (ry2 - ry1) >= 8:
                infer_x1, infer_y1 = rx1, ry1
                infer_frame = frame[ry1:ry2, rx1:rx2]

        t0 = time.perf_counter()
        track_kwargs = self._runtime_kwargs()
        track_results = self.model.track(
            infer_frame,
            conf=self.model_cfg["conf_threshold"],
            persist=True,
            verbose=False,
            **track_kwargs,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0

        detections = self._extract_track_detections(
            result=track_results[0],
            frame_shape=frame.shape,
            x_offset=infer_x1,
            y_offset=infer_y1,
        )

        if self.fallback_model is not None:
            fallback_kwargs = {
                "imgsz": self.fallback_imgsz,
                "verbose": False,
            }
            if self.device:
                fallback_kwargs["device"] = self.device
            if self.use_half:
                fallback_kwargs["half"] = True
            fallback_results = self.fallback_model.predict(
                infer_frame,
                conf=self.model_cfg["person_fallback_conf"],
                classes=[0],
                **fallback_kwargs,
            )
            fallback_dets = self._extract_fallback_person_detections(
                result=fallback_results[0],
                frame_shape=frame.shape,
                x_offset=infer_x1,
                y_offset=infer_y1,
            )
            detections.extend(self._dedup_fallback_persons(detections, fallback_dets))

        if bool(self.model_cfg.get("debug_log_infer", False)):
            by_name: dict[str, int] = {}
            for d in detections:
                by_name[d.cls_name] = by_name.get(d.cls_name, 0) + 1
            print(f"[infer-main] detections={len(detections)} by_class={by_name}", flush=True)

        return detections, infer_ms

    def _infer_main_binary(
        self,
        frame,
        roi_rect: tuple[int, int, int, int],
    ) -> tuple[list[Detection], float]:
        infer_x1, infer_y1 = 0, 0
        infer_frame = frame
        if self.roi_cfg["enabled"] and self.roi_cfg.get("global_inference_in_roi", False):
            rx1, ry1, rx2, ry2 = roi_rect
            if (rx2 - rx1) >= 8 and (ry2 - ry1) >= 8:
                infer_x1, infer_y1 = rx1, ry1
                infer_frame = frame[ry1:ry2, rx1:rx2]

        t0 = time.perf_counter()
        predict_kwargs = {
            "imgsz": self.binary_imgsz,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device:
            predict_kwargs["device"] = self.device
        if self.use_half:
            predict_kwargs["half"] = True
        pred_results = self.model.predict(
            infer_frame,
            conf=self.binary_conf_threshold,
            **predict_kwargs,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        detections = self._extract_binary_detections(
            result=pred_results[0],
            frame_shape=frame.shape,
            x_offset=infer_x1,
            y_offset=infer_y1,
        )
        return detections, infer_ms

    def _extract_track_detections(
        self,
        result,
        frame_shape,
        x_offset: int,
        y_offset: int,
    ) -> list[Detection]:
        boxes_obj = result.boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return []

        xyxy = boxes_obj.xyxy.cpu().tolist()
        clss = boxes_obj.cls.int().cpu().tolist()
        confs = boxes_obj.conf.cpu().tolist()
        if boxes_obj.id is not None:
            track_ids = boxes_obj.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(xyxy)

        frame_h, frame_w = frame_shape[:2]
        detections: list[Detection] = []
        for (lx1, ly1, lx2, ly2), cls_id, conf, track_id in zip(xyxy, clss, confs, track_ids):
            cls_id = int(cls_id)
            raw_cls_name = self.id_to_name.get(cls_id, str(cls_id))
            cls_name = self._normalize_class_name(raw_cls_name)
            if cls_name not in {"person", "head", "hardhat", "vest", "no_vest"}:
                continue
            gx1, gy1, gx2, gy2 = clip_xyxy_to_frame(
                float(lx1 + x_offset),
                float(ly1 + y_offset),
                float(lx2 + x_offset),
                float(ly2 + y_offset),
                frame_w,
                frame_h,
            )
            detections.append(
                Detection(
                    cls_id=self.class_ids[cls_name],
                    cls_name=cls_name,
                    conf=float(conf),
                    bbox_xyxy=(gx1, gy1, gx2, gy2),
                    track_id=int(track_id) if track_id is not None else None,
                    source="main_track",
                )
            )
        return detections

    def _extract_fallback_person_detections(
        self,
        result,
        frame_shape,
        x_offset: int,
        y_offset: int,
    ) -> list[Detection]:
        boxes_obj = result.boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return []
        xyxy = boxes_obj.xyxy.cpu().tolist()
        confs = boxes_obj.conf.cpu().tolist()

        frame_h, frame_w = frame_shape[:2]
        fallback_dets: list[Detection] = []
        for (lx1, ly1, lx2, ly2), conf in zip(xyxy, confs):
            gx1, gy1, gx2, gy2 = clip_xyxy_to_frame(
                float(lx1 + x_offset),
                float(ly1 + y_offset),
                float(lx2 + x_offset),
                float(ly2 + y_offset),
                frame_w,
                frame_h,
            )
            fallback_dets.append(
                Detection(
                    cls_id=self.class_ids["person"],
                    cls_name="person",
                    conf=float(conf),
                    bbox_xyxy=(gx1, gy1, gx2, gy2),
                    track_id=None,
                    source="fallback_person",
                )
            )
        return fallback_dets

    def _extract_binary_detections(
        self,
        result,
        frame_shape,
        x_offset: int,
        y_offset: int,
    ) -> list[Detection]:
        boxes_obj = result.boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return []
        xyxy = boxes_obj.xyxy.cpu().tolist()
        clss = boxes_obj.cls.int().cpu().tolist()
        confs = boxes_obj.conf.cpu().tolist()
        frame_h, frame_w = frame_shape[:2]
        detections: list[Detection] = []
        for (lx1, ly1, lx2, ly2), cls_id, conf in zip(xyxy, clss, confs):
            cls_name = self._normalize_class_name(self.id_to_name.get(int(cls_id), str(cls_id)))
            cls_norm = str(cls_name).strip().lower().replace("-", "_")
            state = None
            if cls_norm in self.binary_with_aliases:
                state = "with_hard_hat"
            elif cls_norm in self.binary_without_aliases:
                state = "without_hard_hat"
            if state is None:
                continue
            gx1, gy1, gx2, gy2 = clip_xyxy_to_frame(
                float(lx1 + x_offset),
                float(ly1 + y_offset),
                float(lx2 + x_offset),
                float(ly2 + y_offset),
                frame_w,
                frame_h,
            )
            detections.append(
                Detection(
                    cls_id=self.class_ids["person"],
                    cls_name="person",
                    conf=float(conf),
                    bbox_xyxy=(gx1, gy1, gx2, gy2),
                    track_id=None,
                    source="binary_hardhat",
                    hardhat_state=state,
                )
            )
        return detections

    @staticmethod
    def _dedup_fallback_persons(main_dets: list[Detection], fallback_dets: list[Detection]) -> list[Detection]:
        existing_person_boxes = [d.bbox_xyxy for d in main_dets if d.cls_name == "person"]
        accepted: list[Detection] = []
        for det in fallback_dets:
            if any(bbox_iou(det.bbox_xyxy, existing) >= 0.45 for existing in existing_person_boxes):
                continue
            existing_person_boxes.append(det.bbox_xyxy)
            accepted.append(det)
        return accepted

    def infer_head_hardhat_in_person_rois(
        self,
        frame,
        person_boxes: dict[int, tuple[float, float, float, float]],
    ) -> list[Detection]:
        if self.binary_enabled or self.roi_model is None:
            return []
        frame_h, frame_w = frame.shape[:2]
        detections: list[Detection] = []
        for person_id, (px1, py1, px2, py2) in person_boxes.items():
            pw = px2 - px1
            ph = py2 - py1
            rx1 = int(max(0.0, px1 - self.person_roi_cfg["x_expand_ratio"] * pw))
            ry1 = int(max(0.0, py1 - self.person_roi_cfg["y_expand_top_ratio"] * ph))
            rx2 = int(min(float(frame_w), px2 + self.person_roi_cfg["x_expand_ratio"] * pw))
            ry2 = int(min(float(frame_h), py1 + self.person_roi_cfg["y_bottom_ratio"] * ph))
            if rx2 - rx1 < 8 or ry2 - ry1 < 8:
                continue

            person_roi = frame[ry1:ry2, rx1:rx2]
            if person_roi.size == 0:
                continue

            roi_classes = [cid for cid in (self.class_ids.get("head", -1), self.class_ids.get("hardhat", -1)) if cid >= 0]
            if not roi_classes:
                continue
            roi_results = self.roi_model.predict(
                person_roi,
                conf=self.model_cfg["roi_head_hardhat_conf"],
                classes=roi_classes,
                imgsz=self.roi_imgsz,
                verbose=False,
                **({"device": self.device} if self.device else {}),
                **({"half": True} if self.use_half else {}),
            )
            roi_boxes_obj = roi_results[0].boxes
            if roi_boxes_obj is None or len(roi_boxes_obj) == 0:
                continue

            roi_xyxy = roi_boxes_obj.xyxy.cpu().tolist()
            roi_clss = roi_boxes_obj.cls.int().cpu().tolist()
            roi_confs = roi_boxes_obj.conf.cpu().tolist()

            for (bx1, by1, bx2, by2), cls_id, conf in zip(roi_xyxy, roi_clss, roi_confs):
                cls_id = int(cls_id)
                cls_name = self._normalize_class_name(self.id_to_name.get(cls_id, str(cls_id)))
                if cls_name not in {"head", "hardhat"}:
                    continue
                gx1, gy1, gx2, gy2 = clip_xyxy_to_frame(
                    float(bx1 + rx1),
                    float(by1 + ry1),
                    float(bx2 + rx1),
                    float(by2 + ry1),
                    frame_w,
                    frame_h,
                )
                detections.append(
                    Detection(
                        cls_id=self.class_ids[cls_name],
                        cls_name=cls_name,
                        conf=float(conf),
                        bbox_xyxy=(gx1, gy1, gx2, gy2),
                        track_id=None,
                        source="person_roi",
                        owner_person_id=person_id,
                    )
                )
        return detections
