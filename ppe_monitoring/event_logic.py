from __future__ import annotations

from collections import defaultdict

from .types import ViolationEvent


class TemporalEventLogic:
    def __init__(self, cfg: dict):
        event_cfg = cfg["event_logic"]
        self.confirm_frames = int(event_cfg.get("hardhat_confirm_frames", 2))
        self.revoke_frames = int(event_cfg.get("hardhat_revoke_frames", 4))
        self.lock_after_confirm = bool(event_cfg.get("lock_after_confirm", True))
        self.hardhat_confirm_seconds = float(event_cfg.get("hardhat_confirm_seconds", 0.0))
        self.hardhat_ever_locks_status = bool(event_cfg.get("hardhat_ever_locks_status", True))
        self.no_hardhat_consecutive_frames = int(event_cfg.get("no_hardhat_consecutive_frames", 20))
        self.no_hardhat_seconds_threshold = float(event_cfg.get("no_hardhat_seconds_threshold", 2.0))
        self.cooldown_frames = int(event_cfg.get("cooldown_frames", 90))
        self.cooldown_seconds = float(event_cfg.get("cooldown_seconds", 3.0))
        self.max_no_hardhat_hold_frames_without_infer = int(
            event_cfg.get("max_no_hardhat_hold_frames_without_infer", 4)
        )
        self.no_vest_consecutive_frames = int(event_cfg.get("no_vest_consecutive_frames", 12))
        self.no_vest_seconds_threshold = float(event_cfg.get("no_vest_seconds_threshold", 1.5))
        self.max_no_vest_hold_frames_without_infer = int(
            event_cfg.get("max_no_vest_hold_frames_without_infer", 4)
        )
        # Production-like: one alert per continuous violation episode per track (reset when violation ends).
        self.one_event_per_active_episode = bool(event_cfg.get("one_event_per_active_episode", False))
        self.one_vest_event_per_active_episode = bool(
            event_cfg.get("one_vest_event_per_active_episode", self.one_event_per_active_episode)
        )

        self.hardhat_confirmed = defaultdict(lambda: False)
        self.hardhat_positive_streak = defaultdict(int)
        self.hardhat_miss_streak = defaultdict(int)
        self.no_hardhat_streak = defaultdict(int)
        self.last_hardhat_seen_ts: dict[int, float] = {}
        self.last_infer_frame: dict[int, int] = {}
        self.violation_active_state = defaultdict(lambda: False)
        self.last_alert_frame = defaultdict(lambda: -10**9)
        self.last_alert_ts = defaultdict(lambda: -10**9)
        self.first_seen_frame: dict[int, int] = {}
        self.first_seen_ts: dict[int, float] = {}
        self.event_counter = 0
        self.has_hardhat_ever: set[int] = set()
        self.hardhat_positive_start_ts: dict[int, float] = {}
        self.active_no_hardhat_persons: set[int] = set()
        self.unique_no_hardhat_persons: set[int] = set()
        self.no_vest_streak = defaultdict(int)
        self.last_vest_seen_ts: dict[int, float] = {}
        self.vest_violation_active_state = defaultdict(lambda: False)
        self.last_vest_alert_frame = defaultdict(lambda: -10**9)
        self.last_vest_alert_ts = defaultdict(lambda: -10**9)
        self.active_no_vest_persons: set[int] = set()
        self.unique_no_vest_persons: set[int] = set()
        self.no_hardhat_episode_emitted = defaultdict(lambda: False)
        self.no_vest_episode_emitted = defaultdict(lambda: False)

    def update(
        self,
        person_boxes: dict[int, tuple[float, float, float, float]],
        person_hardhat_observed: dict[int, bool],
        person_vest_observed: dict[int, bool],
        frame_idx: int,
        timestamp_sec: float,
        did_infer: bool,
    ) -> tuple[dict[int, bool], list[ViolationEvent], int, set[int]]:
        statuses: dict[int, bool] = {}
        events: list[ViolationEvent] = []
        violating_person_ids: set[int] = set()
        violating_no_hardhat_ids: set[int] = set()
        violating_no_vest_ids: set[int] = set()

        for person_id in person_boxes:
            if person_id not in self.first_seen_frame:
                self.first_seen_frame[person_id] = frame_idx
                self.first_seen_ts[person_id] = timestamp_sec

            if person_id not in self.last_hardhat_seen_ts:
                self.last_hardhat_seen_ts[person_id] = self.first_seen_ts[person_id]
            if person_id not in self.last_vest_seen_ts:
                self.last_vest_seen_ts[person_id] = self.first_seen_ts[person_id]

            if did_infer:
                self.last_infer_frame[person_id] = frame_idx
                observed = bool(person_hardhat_observed.get(person_id, False))
                if observed:
                    self.has_hardhat_ever.add(person_id)
                    if person_id not in self.hardhat_positive_start_ts:
                        self.hardhat_positive_start_ts[person_id] = timestamp_sec
                    self.hardhat_positive_streak[person_id] += 1
                    self.hardhat_miss_streak[person_id] = 0
                    self.last_hardhat_seen_ts[person_id] = timestamp_sec
                else:
                    self.hardhat_positive_start_ts.pop(person_id, None)
                    self.hardhat_positive_streak[person_id] = 0
                    self.hardhat_miss_streak[person_id] += 1

                positive_started_ts = self.hardhat_positive_start_ts.get(person_id, timestamp_sec)
                positive_duration_sec = max(0.0, timestamp_sec - positive_started_ts)
                meets_confirm_seconds = positive_duration_sec >= self.hardhat_confirm_seconds
                if (
                    not self.hardhat_confirmed[person_id]
                    and self.hardhat_positive_streak[person_id] >= self.confirm_frames
                    and meets_confirm_seconds
                ):
                    self.hardhat_confirmed[person_id] = True
                if (
                    not self.lock_after_confirm
                    and self.hardhat_confirmed[person_id]
                    and self.hardhat_miss_streak[person_id] >= self.revoke_frames
                ):
                    self.hardhat_confirmed[person_id] = False

            if self.hardhat_ever_locks_status:
                has_hardhat = bool(person_id in self.has_hardhat_ever) or (
                    bool(self.hardhat_confirmed[person_id]) and (self.hardhat_miss_streak[person_id] < self.revoke_frames)
                )
            else:
                has_hardhat = bool(self.hardhat_confirmed[person_id]) and (
                    self.hardhat_miss_streak[person_id] < self.revoke_frames
                )
            statuses[person_id] = has_hardhat

            if did_infer:
                if has_hardhat:
                    self.no_hardhat_streak[person_id] = 0
                else:
                    self.no_hardhat_streak[person_id] += 1

            no_hardhat_duration = max(0.0, timestamp_sec - self.last_hardhat_seen_ts[person_id])
            meets_frames = self.no_hardhat_streak[person_id] >= self.no_hardhat_consecutive_frames
            meets_seconds = no_hardhat_duration >= self.no_hardhat_seconds_threshold
            if person_id in self.has_hardhat_ever:
                self.no_hardhat_streak[person_id] = 0
                self.violation_active_state[person_id] = False
            elif did_infer:
                self.violation_active_state[person_id] = (not has_hardhat) and meets_frames and meets_seconds
            elif (frame_idx - self.last_infer_frame.get(person_id, -10**9)) > self.max_no_hardhat_hold_frames_without_infer:
                self.violation_active_state[person_id] = False
            no_hardhat_violation_active = bool(self.violation_active_state[person_id])
            if not no_hardhat_violation_active:
                self.no_hardhat_episode_emitted[person_id] = False

            if no_hardhat_violation_active:
                violating_person_ids.add(person_id)
                violating_no_hardhat_ids.add(person_id)
                self.unique_no_hardhat_persons.add(person_id)
                cooldown_by_frames = (frame_idx - self.last_alert_frame[person_id]) >= self.cooldown_frames
                cooldown_by_time = (timestamp_sec - self.last_alert_ts[person_id]) >= self.cooldown_seconds
                may_emit = cooldown_by_frames and cooldown_by_time
                if self.one_event_per_active_episode:
                    may_emit = may_emit and not self.no_hardhat_episode_emitted[person_id]
                if may_emit:
                    self.event_counter += 1
                    pbox = person_boxes.get(person_id)
                    event = ViolationEvent(
                        event_id=self.event_counter,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        person_track_id=person_id,
                        event_type="no_hardhat",
                        no_hardhat_streak=self.no_hardhat_streak[person_id],
                        no_hardhat_duration_sec=no_hardhat_duration,
                        person_bbox=tuple(float(v) for v in pbox) if pbox is not None else None,
                    )
                    events.append(event)
                    self.last_alert_frame[person_id] = frame_idx
                    self.last_alert_ts[person_id] = timestamp_sec
                    if self.one_event_per_active_episode:
                        self.no_hardhat_episode_emitted[person_id] = True

            if did_infer:
                vest_observed = bool(person_vest_observed.get(person_id, False))
                if vest_observed:
                    self.no_vest_streak[person_id] = 0
                    self.last_vest_seen_ts[person_id] = timestamp_sec
                else:
                    self.no_vest_streak[person_id] += 1
            no_vest_duration = max(0.0, timestamp_sec - self.last_vest_seen_ts[person_id])
            vest_meets_frames = self.no_vest_streak[person_id] >= self.no_vest_consecutive_frames
            vest_meets_seconds = no_vest_duration >= self.no_vest_seconds_threshold
            if did_infer:
                self.vest_violation_active_state[person_id] = vest_meets_frames and vest_meets_seconds
            elif (frame_idx - self.last_infer_frame.get(person_id, -10**9)) > self.max_no_vest_hold_frames_without_infer:
                self.vest_violation_active_state[person_id] = False
            no_vest_violation_active = bool(self.vest_violation_active_state[person_id])
            if not no_vest_violation_active:
                self.no_vest_episode_emitted[person_id] = False
            if no_vest_violation_active:
                violating_person_ids.add(person_id)
                violating_no_vest_ids.add(person_id)
                self.unique_no_vest_persons.add(person_id)
                cooldown_by_frames = (frame_idx - self.last_vest_alert_frame[person_id]) >= self.cooldown_frames
                cooldown_by_time = (timestamp_sec - self.last_vest_alert_ts[person_id]) >= self.cooldown_seconds
                may_emit_vest = cooldown_by_frames and cooldown_by_time
                if self.one_vest_event_per_active_episode:
                    may_emit_vest = may_emit_vest and not self.no_vest_episode_emitted[person_id]
                if may_emit_vest:
                    self.event_counter += 1
                    pbox = person_boxes.get(person_id)
                    events.append(
                        ViolationEvent(
                            event_id=self.event_counter,
                            frame_idx=frame_idx,
                            timestamp_sec=timestamp_sec,
                            person_track_id=person_id,
                            event_type="no_vest",
                            no_hardhat_streak=self.no_vest_streak[person_id],
                            no_hardhat_duration_sec=no_vest_duration,
                            person_bbox=tuple(float(v) for v in pbox) if pbox is not None else None,
                        )
                    )
                    self.last_vest_alert_frame[person_id] = frame_idx
                    self.last_vest_alert_ts[person_id] = timestamp_sec
                    if self.one_vest_event_per_active_episode:
                        self.no_vest_episode_emitted[person_id] = True

        self.active_no_hardhat_persons = set(violating_no_hardhat_ids)
        self.active_no_vest_persons = set(violating_no_vest_ids)
        active_violations = len(violating_person_ids)
        return statuses, events, active_violations, violating_person_ids

    def seen_seconds(self, person_id: int, frame_idx: int, input_fps: float) -> float:
        if person_id not in self.first_seen_frame:
            return 0.0
        seen_frames = frame_idx - self.first_seen_frame[person_id] + 1
        fps = input_fps if input_fps > 0 else 30.0
        return seen_frames / float(fps)

    def remove_ids(self, person_ids: list[int]) -> None:
        for person_id in person_ids:
            self.hardhat_confirmed.pop(person_id, None)
            self.hardhat_positive_streak.pop(person_id, None)
            self.hardhat_miss_streak.pop(person_id, None)
            self.no_hardhat_streak.pop(person_id, None)
            self.last_hardhat_seen_ts.pop(person_id, None)
            self.last_vest_seen_ts.pop(person_id, None)
            self.last_infer_frame.pop(person_id, None)
            self.violation_active_state.pop(person_id, None)
            self.vest_violation_active_state.pop(person_id, None)
            self.last_alert_frame.pop(person_id, None)
            self.last_alert_ts.pop(person_id, None)
            self.last_vest_alert_frame.pop(person_id, None)
            self.last_vest_alert_ts.pop(person_id, None)
            self.first_seen_frame.pop(person_id, None)
            self.first_seen_ts.pop(person_id, None)
            self.has_hardhat_ever.discard(person_id)
            self.hardhat_positive_start_ts.pop(person_id, None)
            self.active_no_hardhat_persons.discard(person_id)
            self.active_no_vest_persons.discard(person_id)
            self.no_vest_streak.pop(person_id, None)
            self.no_hardhat_episode_emitted.pop(person_id, None)
            self.no_vest_episode_emitted.pop(person_id, None)
