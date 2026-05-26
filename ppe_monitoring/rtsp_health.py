from __future__ import annotations

import time


class RtspHealthWatchdog:
    """Collects RTSP connectivity/read health metrics for deployment diagnostics."""

    def __init__(self, is_rtsp: bool):
        self.is_rtsp = bool(is_rtsp)

        self.open_attempts_total = 0
        self.open_successes = 0
        self.open_failures = 0

        self.frames_read_ok = 0
        self.read_failures_total = 0
        self.consecutive_read_failures = 0
        self.max_consecutive_read_failures = 0

        self.disconnect_events = 0
        self.reconnect_cycles = 0
        self.reconnect_attempts_total = 0
        self.reconnect_successes = 0
        self.reconnect_failures = 0

        self.total_downtime_sec = 0.0
        self.longest_downtime_sec = 0.0
        self._disconnect_started_at = None

    def on_open_result(self, success: bool) -> None:
        self.open_attempts_total += 1
        if success:
            self.open_successes += 1
        else:
            self.open_failures += 1

    def on_frame_read_success(self) -> None:
        self.frames_read_ok += 1
        self.consecutive_read_failures = 0

    def on_frame_read_failure(self) -> None:
        self.read_failures_total += 1
        self.consecutive_read_failures += 1
        self.max_consecutive_read_failures = max(
            self.max_consecutive_read_failures,
            self.consecutive_read_failures,
        )

    def on_disconnect(self) -> None:
        if not self.is_rtsp:
            return
        self.disconnect_events += 1
        if self._disconnect_started_at is None:
            self._disconnect_started_at = time.perf_counter()

    def on_reconnect_cycle_start(self) -> None:
        if not self.is_rtsp:
            return
        self.reconnect_cycles += 1

    def on_reconnect_attempt(self) -> None:
        if not self.is_rtsp:
            return
        self.reconnect_attempts_total += 1

    def on_reconnect_success(self) -> None:
        if not self.is_rtsp:
            return
        self.reconnect_successes += 1
        self._close_downtime_window()

    def on_reconnect_cycle_failed(self) -> None:
        if not self.is_rtsp:
            return
        self.reconnect_failures += 1
        self._close_downtime_window()

    def _close_downtime_window(self) -> None:
        if self._disconnect_started_at is None:
            return
        downtime = max(0.0, time.perf_counter() - self._disconnect_started_at)
        self.total_downtime_sec += downtime
        self.longest_downtime_sec = max(self.longest_downtime_sec, downtime)
        self._disconnect_started_at = None

    def finalize(self) -> None:
        # Close unclosed downtime period if stream ended while disconnected.
        self._close_downtime_window()

    def to_dict(self) -> dict:
        return {
            "is_rtsp": self.is_rtsp,
            "open_attempts_total": self.open_attempts_total,
            "open_successes": self.open_successes,
            "open_failures": self.open_failures,
            "frames_read_ok": self.frames_read_ok,
            "read_failures_total": self.read_failures_total,
            "consecutive_read_failures": self.consecutive_read_failures,
            "max_consecutive_read_failures": self.max_consecutive_read_failures,
            "disconnect_events": self.disconnect_events,
            "reconnect_cycles": self.reconnect_cycles,
            "reconnect_attempts_total": self.reconnect_attempts_total,
            "reconnect_successes": self.reconnect_successes,
            "reconnect_failures": self.reconnect_failures,
            "total_downtime_sec": round(self.total_downtime_sec, 3),
            "longest_downtime_sec": round(self.longest_downtime_sec, 3),
        }
