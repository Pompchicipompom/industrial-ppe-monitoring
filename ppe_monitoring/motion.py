from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", True))
        self.pixel_threshold = int(cfg.get("pixel_threshold", 25))
        self.blur_kernel = int(cfg.get("blur_kernel", 5))
        self.background_alpha = float(cfg.get("background_alpha", 0.03))
        self.use_morphology = bool(cfg.get("use_morphology", True))
        self.morph_kernel = int(cfg.get("morph_kernel", 3))
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        if self.morph_kernel % 2 == 0:
            self.morph_kernel += 1
        self.background_gray = None

    def update(self, frame_bgr) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        if self.background_gray is None:
            self.background_gray = gray.astype(np.float32)
            return 0.0

        bg_u8 = cv2.convertScaleAbs(self.background_gray)
        diff = cv2.absdiff(bg_u8, gray)
        _, binary = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        if self.use_morphology and self.morph_kernel > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        changed = cv2.countNonZero(binary)
        total = binary.shape[0] * binary.shape[1]

        cv2.accumulateWeighted(gray, self.background_gray, self.background_alpha)
        return changed / float(total) if total > 0 else 0.0


class FrameSampler:
    def __init__(self, target_fps: float):
        self.target_fps = float(target_fps)
        self.period_sec = 0.0 if self.target_fps <= 0 else 1.0 / self.target_fps
        self.last_sample_ts = -1e9

    def should_sample(self, timestamp_sec: float) -> bool:
        if self.period_sec <= 0:
            return True
        if (timestamp_sec - self.last_sample_ts) >= self.period_sec:
            self.last_sample_ts = timestamp_sec
            return True
        return False


class InferenceGate:
    def __init__(self, pipeline_cfg: dict, motion_cfg: dict):
        self.mode = str(pipeline_cfg.get("mode", "motion_gated"))
        self.force_infer_every_n_frames = int(pipeline_cfg.get("force_infer_every_n_frames", 30))
        self.motion_enabled = bool(motion_cfg.get("enabled", True))
        self.motion_min_ratio = float(motion_cfg.get("min_ratio", 0.0025))
        self.motion_min_ratio_off = float(
            motion_cfg.get("min_ratio_off", max(0.0, self.motion_min_ratio * 0.6))
        )
        self.motion_hold_frames = int(motion_cfg.get("hold_frames_after_motion", 8))
        self.motion_active = False
        self.motion_hold_until_frame = -10**9

    def should_infer(
        self,
        frame_idx: int,
        sampled: bool,
        motion_ratio: float,
        last_infer_frame: int,
    ) -> bool:
        due_to_force = (frame_idx - last_infer_frame) >= self.force_infer_every_n_frames
        if self.mode == "every_sample":
            return sampled or due_to_force
        if not self.motion_enabled:
            return sampled or due_to_force

        if motion_ratio >= self.motion_min_ratio:
            self.motion_active = True
            self.motion_hold_until_frame = frame_idx + self.motion_hold_frames
        elif motion_ratio < self.motion_min_ratio_off and frame_idx > self.motion_hold_until_frame:
            self.motion_active = False

        motion_signal = self.motion_active or (frame_idx <= self.motion_hold_until_frame)
        return (sampled and motion_signal) or due_to_force
