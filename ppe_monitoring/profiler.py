from __future__ import annotations

import time
from collections import defaultdict, deque

import numpy as np

# Cap stored per-frame samples to bound memory on long RTSP sessions (~8 bytes * N per series).
_DEFAULT_MAX_SAMPLES = 200_000


class RuntimeProfiler:
    def __init__(self, max_samples: int | None = None):
        self.max_samples = max(1000, int(max_samples) if max_samples is not None else _DEFAULT_MAX_SAMPLES)
        self.start_time = time.perf_counter()
        self.frame_count = 0
        self.infer_count = 0
        self.infer_ms_samples: deque[float] = deque(maxlen=self.max_samples)
        self.loop_ms_samples: deque[float] = deque(maxlen=self.max_samples)
        self.motion_ratio_samples: deque[float] = deque(maxlen=self.max_samples)
        self.stage_samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.max_samples))

    def record_frame(
        self,
        loop_ms: float,
        did_infer: bool,
        infer_ms: float,
        motion_ratio: float,
        stage_timings_ms: dict[str, float] | None = None,
    ) -> None:
        self.frame_count += 1
        self.loop_ms_samples.append(float(loop_ms))
        self.motion_ratio_samples.append(float(motion_ratio))
        if did_infer:
            self.infer_count += 1
            self.infer_ms_samples.append(float(infer_ms))
        if stage_timings_ms:
            for stage_name, value in stage_timings_ms.items():
                self.stage_samples[str(stage_name)].append(float(value))

    def _stats(self, values: deque[float] | list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0, "sum": 0.0, "count": 0.0}
        arr = np.array(list(values), dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "sum": float(arr.sum()),
            "count": float(arr.size),
        }

    def _stage_summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for stage_name in sorted(self.stage_samples.keys()):
            st = self._stats(self.stage_samples[stage_name])
            out[stage_name] = {
                "mean": st["mean"],
                "median": st["median"],
                "p90": st["p90"],
                "sum": st["sum"],
                "count": st["count"],
            }
        return out

    def summary(self, input_fps: float) -> dict[str, float]:
        elapsed_sec = max(time.perf_counter() - self.start_time, 1e-6)
        processing_fps = self.frame_count / elapsed_sec
        inference_fps = self.infer_count / elapsed_sec
        infer_stats = self._stats(self.infer_ms_samples)
        loop_stats = self._stats(self.loop_ms_samples)
        ring_truncated = self.frame_count > self.max_samples
        return {
            "elapsed_sec": elapsed_sec,
            "input_fps": float(input_fps),
            "processing_fps": processing_fps,
            "inference_fps": inference_fps,
            "inference_ms_mean": infer_stats["mean"],
            "inference_ms_median": infer_stats["median"],
            "inference_ms_p90": infer_stats["p90"],
            "loop_ms_mean": loop_stats["mean"],
            "loop_ms_median": loop_stats["median"],
            "loop_ms_p90": loop_stats["p90"],
            "frames_total": float(self.frame_count),
            "frames_inferred": float(self.infer_count),
            "stage_timing_ms": self._stage_summary(),
            "profiler_max_samples": float(self.max_samples),
            "profiler_latency_ring_truncated": ring_truncated,
        }
