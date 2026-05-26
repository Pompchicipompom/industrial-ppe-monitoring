from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_GT_COLUMNS = ["video_id", "event_id", "start_frame", "end_frame", "violation_type"]
REQUIRED_PRED_COLUMNS = ["frame_idx", "event_type"]
RUN_SUMMARY_REQUIRED_COLUMNS = ["config_name", "video_id", "split", "status", "output_dir"]
ALLOWED_SPLITS = {"dev", "test", "stress", "negative"}


@dataclass
class GTEvent:
    video_id: str
    event_id: str
    start_frame: int
    end_frame: int
    violation_type: str
    zone_id: str = ""
    notes: str = ""


@dataclass
class PredEvent:
    pred_idx: int
    frame_idx: int
    event_type: str
    timestamp_sec: float | None = None
    person_track_id: int | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_run_group_dir(run_group_arg: str, experiments_root: Path, repo_root: Path) -> Path:
    candidate = Path(run_group_arg)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    if candidate.exists():
        return candidate.resolve()

    candidate_under_root = (experiments_root / run_group_arg).resolve()
    if candidate_under_root.exists():
        return candidate_under_root

    raise FileNotFoundError(
        f"Run group directory not found. Checked: '{run_group_arg}' and '{candidate_under_root}'"
    )


def read_runs_summary(run_group_dir: Path) -> list[dict[str, str]]:
    csv_path = run_group_dir / "runs_summary.csv"
    jsonl_path = run_group_dir / "runs_summary.jsonl"

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"runs_summary.csv has no header: {csv_path}")
            missing = [c for c in RUN_SUMMARY_REQUIRED_COLUMNS if c not in set(reader.fieldnames)]
            if missing:
                raise ValueError(f"runs_summary.csv missing required columns {missing}: {csv_path}")
            return list(reader)

    if jsonl_path.exists():
        rows: list[dict[str, str]] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except Exception as exc:
                    raise ValueError(f"Invalid JSONL line {line_idx} in {jsonl_path}") from exc
                rows.append({k: str(v) if v is not None else "" for k, v in obj.items()})
        if not rows:
            raise ValueError(f"runs_summary.jsonl is empty: {jsonl_path}")
        for col in RUN_SUMMARY_REQUIRED_COLUMNS:
            if col not in rows[0]:
                raise ValueError(f"runs_summary.jsonl missing required field '{col}': {jsonl_path}")
        return rows

    raise FileNotFoundError(f"Neither runs_summary.csv nor runs_summary.jsonl found in {run_group_dir}")


def read_gt_events(gt_dir: Path) -> dict[str, list[GTEvent]]:
    if not gt_dir.exists() or not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    files = sorted(p for p in gt_dir.glob("*.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No GT CSV files found in: {gt_dir}")

    out: dict[str, list[GTEvent]] = {}
    seen_event_keys: set[tuple[str, str]] = set()
    for gt_path in files:
        with gt_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"GT file has no header: {gt_path}")
            missing = [c for c in REQUIRED_GT_COLUMNS if c not in set(reader.fieldnames)]
            if missing:
                raise ValueError(f"GT file missing required columns {missing}: {gt_path}")

            for line_idx, row in enumerate(reader, start=2):
                video_id = (row.get("video_id") or "").strip()
                event_id = (row.get("event_id") or "").strip()
                start_raw = (row.get("start_frame") or "").strip()
                end_raw = (row.get("end_frame") or "").strip()
                violation_type = (row.get("violation_type") or "").strip()
                zone_id = (row.get("zone_id") or "").strip() if "zone_id" in reader.fieldnames else ""
                notes = (row.get("notes") or "").strip() if "notes" in reader.fieldnames else ""

                if not video_id:
                    raise ValueError(f"{gt_path}:{line_idx} empty video_id")
                if not event_id:
                    raise ValueError(f"{gt_path}:{line_idx} empty event_id")
                if not start_raw.isdigit():
                    raise ValueError(f"{gt_path}:{line_idx} invalid start_frame '{start_raw}'")
                if not end_raw.isdigit():
                    raise ValueError(f"{gt_path}:{line_idx} invalid end_frame '{end_raw}'")
                start_frame = int(start_raw)
                end_frame = int(end_raw)
                if end_frame < start_frame:
                    raise ValueError(
                        f"{gt_path}:{line_idx} invalid interval end_frame({end_frame}) < start_frame({start_frame})"
                    )
                if not violation_type:
                    raise ValueError(f"{gt_path}:{line_idx} empty violation_type")

                key = (video_id, event_id)
                if key in seen_event_keys:
                    raise ValueError(f"{gt_path}:{line_idx} duplicate (video_id,event_id)={key} across GT files")
                seen_event_keys.add(key)

                event = GTEvent(
                    video_id=video_id,
                    event_id=event_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    violation_type=violation_type,
                    zone_id=zone_id,
                    notes=notes,
                )
                out.setdefault(video_id, []).append(event)

    for video_id in out:
        out[video_id].sort(key=lambda e: (e.start_frame, e.end_frame, e.event_id))
    return out


def read_pred_events(events_csv_path: Path) -> list[PredEvent]:
    if not events_csv_path.exists():
        return []
    with events_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        missing = [c for c in REQUIRED_PRED_COLUMNS if c not in set(reader.fieldnames)]
        if missing:
            raise ValueError(f"Prediction file missing required columns {missing}: {events_csv_path}")

        preds: list[PredEvent] = []
        for idx, row in enumerate(reader):
            frame_raw = (row.get("frame_idx") or "").strip()
            event_type = (row.get("event_type") or "").strip()
            ts_raw = (row.get("timestamp_sec") or "").strip()

            if not frame_raw:
                continue
            try:
                frame_idx = int(float(frame_raw))
            except Exception:
                continue
            if not event_type:
                continue
            timestamp_sec: float | None = None
            if ts_raw:
                try:
                    timestamp_sec = float(ts_raw)
                except Exception:
                    timestamp_sec = None
            track_raw = (row.get("person_track_id") or "").strip()
            person_track_id: int | None = None
            if track_raw:
                try:
                    person_track_id = int(float(track_raw))
                except Exception:
                    person_track_id = None
            preds.append(
                PredEvent(
                    pred_idx=idx,
                    frame_idx=frame_idx,
                    event_type=event_type,
                    timestamp_sec=timestamp_sec,
                    person_track_id=person_track_id,
                )
            )
    preds.sort(key=lambda x: (x.frame_idx, x.pred_idx))
    return preds


def read_input_fps_and_duration_sec(output_dir: Path) -> tuple[float, float]:
    runtime_profile = output_dir / "runtime_profile.json"
    frame_metrics = output_dir / "frame_metrics.csv"

    input_fps = 30.0
    frames_total = None
    if runtime_profile.exists():
        try:
            data = json.loads(runtime_profile.read_text(encoding="utf-8"))
            runtime = data.get("runtime_summary", {})
            fps_val = float(runtime.get("input_fps", 0.0) or 0.0)
            if fps_val > 0:
                input_fps = fps_val
            frames_total_val = runtime.get("frames_total")
            if frames_total_val is not None:
                try:
                    frames_total = float(frames_total_val)
                except Exception:
                    frames_total = None
        except Exception:
            pass

    duration_from_metrics = None
    if frame_metrics.exists():
        try:
            max_ts = None
            with frame_metrics.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is not None and "timestamp_sec" in set(reader.fieldnames):
                    for row in reader:
                        ts_raw = (row.get("timestamp_sec") or "").strip()
                        if not ts_raw:
                            continue
                        try:
                            ts = float(ts_raw)
                        except Exception:
                            continue
                        if max_ts is None or ts > max_ts:
                            max_ts = ts
            if max_ts is not None:
                duration_from_metrics = max_ts + (1.0 / input_fps if input_fps > 0 else 0.0)
        except Exception:
            duration_from_metrics = None

    if duration_from_metrics is not None:
        return input_fps, max(duration_from_metrics, 0.0)

    if frames_total is not None and input_fps > 0:
        return input_fps, max(float(frames_total) / input_fps, 0.0)

    return input_fps, 0.0


def distance_to_interval(frame_idx: int, start: int, end: int) -> int:
    if frame_idx < start:
        return start - frame_idx
    if frame_idx > end:
        return frame_idx - end
    return 0


def match_events(
    preds: list[PredEvent],
    gts: list[GTEvent],
    tolerance_frames: int,
) -> tuple[int, int, int, list[float]]:
    """Match predicted point events to GT intervals (docs/event_evaluation_protocol.md).

    Preconditions (enforced by callers):
    - ``preds`` sorted by ``frame_idx`` ascending (see ``read_pred_events``).
    - One GT matched by at most one pred; extra preds in same interval -> FP.
    - Tie-break among GT candidates: minimal distance to interval, then |pred-start|, then start_frame, event_id.
    """
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must be >= 0")

    matched_gt_indices: set[int] = set()
    tp = 0
    fp = 0
    delays_frames: list[int] = []

    for pred in preds:
        candidate_indices: list[int] = []
        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt_indices:
                continue
            if pred.event_type != gt.violation_type:
                continue
            lo = gt.start_frame - tolerance_frames
            hi = gt.end_frame + tolerance_frames
            if lo <= pred.frame_idx <= hi:
                candidate_indices.append(gt_idx)

        if not candidate_indices:
            fp += 1
            continue

        candidate_indices.sort(
            key=lambda idx: (
                distance_to_interval(pred.frame_idx, gts[idx].start_frame, gts[idx].end_frame),
                abs(pred.frame_idx - gts[idx].start_frame),
                gts[idx].start_frame,
                gts[idx].event_id,
            )
        )
        selected_idx = candidate_indices[0]
        matched_gt_indices.add(selected_idx)
        tp += 1

        gt = gts[selected_idx]
        delay_frames = max(0, pred.frame_idx - gt.start_frame)
        delays_frames.append(delay_frames)

    fn = len(gts) - len(matched_gt_indices)
    return tp, fp, fn, [float(x) for x in delays_frames]


def calc_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def to_float_str(value: float | None, ndigits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{ndigits}f}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_baseline_vs_proposed_rows(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cfg = {row["config_name"]: row for row in aggregate_rows if row.get("eval_status") == "evaluated"}
    baseline = by_cfg.get("baseline")
    proposed = by_cfg.get("proposed")

    metrics = [
        ("videos_evaluated", "videos_evaluated"),
        ("tp_total", "tp_total"),
        ("fp_total", "fp_total"),
        ("fn_total", "fn_total"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "f1"),
        ("false_alarms_per_hour", "false_alarms_per_hour"),
        ("mean_detection_delay_sec", "mean_detection_delay_sec"),
    ]

    rows: list[dict[str, Any]] = []
    for metric_name, key in metrics:
        baseline_val = baseline.get(key, "") if baseline else ""
        proposed_val = proposed.get(key, "") if proposed else ""

        delta = ""
        try:
            if baseline is not None and proposed is not None and str(baseline_val) != "" and str(proposed_val) != "":
                delta = float(proposed_val) - float(baseline_val)
                delta = f"{delta:.6f}"
        except Exception:
            delta = ""

        rows.append(
            {
                "metric": metric_name,
                "baseline": baseline_val,
                "proposed": proposed_val,
                "delta_proposed_minus_baseline": delta,
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Event-level evaluator for baseline/proposed runs under output_files/experiments/{run_group}."
    )
    parser.add_argument(
        "--run-group",
        type=str,
        required=True,
        help="Run group name or full path to run group directory.",
    )
    parser.add_argument(
        "--experiments-root",
        type=str,
        default="output_files/experiments",
        help="Root directory for experiment run groups.",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="data/gt_events",
        help="Directory containing GT event CSV files.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="Optional comma-separated config_name filter.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="",
        help="Optional comma-separated split filter (dev,test,stress).",
    )
    parser.add_argument(
        "--tolerance-frames",
        type=int,
        default=0,
        help="Frame tolerance for event matching.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="evaluation",
        help="Subdirectory under run_group for evaluation outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    experiments_root = Path(args.experiments_root)
    if not experiments_root.is_absolute():
        experiments_root = (repo_root / experiments_root).resolve()
    gt_dir = Path(args.gt_dir)
    if not gt_dir.is_absolute():
        gt_dir = (repo_root / gt_dir).resolve()

    run_group_dir = parse_run_group_dir(args.run_group, experiments_root=experiments_root, repo_root=repo_root)
    output_dir = run_group_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_filter = set(parse_csv_list(args.configs))
    split_filter = set(parse_csv_list(args.splits))
    invalid_splits = sorted(s for s in split_filter if s not in ALLOWED_SPLITS)
    if invalid_splits:
        raise ValueError(f"Invalid split filter values: {invalid_splits}. Allowed: {sorted(ALLOWED_SPLITS)}")

    runs = read_runs_summary(run_group_dir)
    gt_map = read_gt_events(gt_dir)

    per_video_rows: list[dict[str, Any]] = []
    aggregate_state: dict[str, dict[str, Any]] = {}

    for run in runs:
        config_name = (run.get("config_name") or "").strip()
        video_id = (run.get("video_id") or "").strip()
        split = (run.get("split") or "").strip()
        status = (run.get("status") or "").strip()
        output_dir_raw = (run.get("output_dir") or "").strip()

        if cfg_filter and config_name not in cfg_filter:
            continue
        if split_filter and split not in split_filter:
            continue

        run_out_dir = Path(output_dir_raw)
        if not run_out_dir.is_absolute():
            run_out_dir = (repo_root / run_out_dir).resolve()

        row: dict[str, Any] = {
            "run_group": run_group_dir.name,
            "config_name": config_name,
            "video_id": video_id,
            "split": split,
            "run_status": status,
            "eval_status": "",
            "tolerance_frames": args.tolerance_frames,
            "gt_events_count": "",
            "pred_events_count": "",
            "tp": "",
            "fp": "",
            "fn": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "input_fps": "",
            "duration_sec": "",
            "false_alarms_per_hour": "",
            "mean_detection_delay_sec": "",
            "output_dir": run.get("output_dir", ""),
            "error_message": "",
        }

        if status != "success":
            row["eval_status"] = f"skipped_status_{status or 'unknown'}"
            per_video_rows.append(row)
            continue

        if not run_out_dir.exists():
            row["eval_status"] = "missing_output_dir"
            row["error_message"] = f"Output directory not found: {run_out_dir}"
            per_video_rows.append(row)
            continue

        events_csv = run_out_dir / "events.csv"
        if not events_csv.exists():
            row["eval_status"] = "missing_events_csv"
            row["error_message"] = f"Prediction file not found: {events_csv}"
            per_video_rows.append(row)
            continue

        try:
            preds = read_pred_events(events_csv)
        except Exception as exc:
            row["eval_status"] = "error_pred_read"
            row["error_message"] = str(exc)
            per_video_rows.append(row)
            continue

        gts = gt_map.get(video_id, [])

        try:
            tp, fp, fn, delays_frames = match_events(preds=preds, gts=gts, tolerance_frames=args.tolerance_frames)
            precision, recall, f1 = calc_prf(tp=tp, fp=fp, fn=fn)
            input_fps, duration_sec = read_input_fps_and_duration_sec(run_out_dir)
        except Exception as exc:
            row["eval_status"] = "error_eval"
            row["error_message"] = str(exc)
            per_video_rows.append(row)
            continue

        fa_per_hour = 0.0
        if duration_sec > 0:
            fa_per_hour = fp / (duration_sec / 3600.0)
        elif fp > 0:
            # Avoid inf in reports while still being explicit via duration column.
            fa_per_hour = 0.0

        if delays_frames:
            mean_delay_sec = sum(d / input_fps for d in delays_frames) / len(delays_frames)
        else:
            mean_delay_sec = None

        row.update(
            {
                "eval_status": "evaluated",
                "gt_events_count": len(gts),
                "pred_events_count": len(preds),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": to_float_str(precision),
                "recall": to_float_str(recall),
                "f1": to_float_str(f1),
                "input_fps": to_float_str(input_fps),
                "duration_sec": to_float_str(duration_sec),
                "false_alarms_per_hour": to_float_str(fa_per_hour),
                "mean_detection_delay_sec": to_float_str(mean_delay_sec) if mean_delay_sec is not None else "",
            }
        )
        per_video_rows.append(row)

        state = aggregate_state.setdefault(
            config_name,
            {
                "config_name": config_name,
                "videos_total": 0,
                "videos_evaluated": 0,
                "videos_skipped": 0,
                "tp_total": 0,
                "fp_total": 0,
                "fn_total": 0,
                "gt_total": 0,
                "pred_total": 0,
                "duration_sec_total": 0.0,
                "delay_sec_sum": 0.0,
                "delay_count": 0,
            },
        )
        state["videos_total"] += 1
        state["videos_evaluated"] += 1
        state["tp_total"] += tp
        state["fp_total"] += fp
        state["fn_total"] += fn
        state["gt_total"] += len(gts)
        state["pred_total"] += len(preds)
        state["duration_sec_total"] += float(duration_sec)
        state["delay_sec_sum"] += sum(d / input_fps for d in delays_frames)
        state["delay_count"] += len(delays_frames)

    for row in per_video_rows:
        if row["eval_status"] != "evaluated":
            cfg = row["config_name"]
            state = aggregate_state.setdefault(
                cfg,
                {
                    "config_name": cfg,
                    "videos_total": 0,
                    "videos_evaluated": 0,
                    "videos_skipped": 0,
                    "tp_total": 0,
                    "fp_total": 0,
                    "fn_total": 0,
                    "gt_total": 0,
                    "pred_total": 0,
                    "duration_sec_total": 0.0,
                    "delay_sec_sum": 0.0,
                    "delay_count": 0,
                },
            )
            state["videos_total"] += 1
            state["videos_skipped"] += 1

    aggregate_rows: list[dict[str, Any]] = []
    for cfg_name in sorted(aggregate_state.keys()):
        st = aggregate_state[cfg_name]
        tp = int(st["tp_total"])
        fp = int(st["fp_total"])
        fn = int(st["fn_total"])
        precision, recall, f1 = calc_prf(tp=tp, fp=fp, fn=fn)

        duration_sec_total = float(st["duration_sec_total"])
        fa_per_hour = 0.0
        if duration_sec_total > 0:
            fa_per_hour = fp / (duration_sec_total / 3600.0)

        mean_delay_sec = None
        if st["delay_count"] > 0:
            mean_delay_sec = float(st["delay_sec_sum"]) / float(st["delay_count"])

        eval_status = "evaluated" if st["videos_evaluated"] > 0 else "no_evaluated_runs"
        aggregate_rows.append(
            {
                "run_group": run_group_dir.name,
                "config_name": cfg_name,
                "eval_status": eval_status,
                "videos_total": st["videos_total"],
                "videos_evaluated": st["videos_evaluated"],
                "videos_skipped": st["videos_skipped"],
                "gt_total": st["gt_total"],
                "pred_total": st["pred_total"],
                "tp_total": tp,
                "fp_total": fp,
                "fn_total": fn,
                "precision": to_float_str(precision),
                "recall": to_float_str(recall),
                "f1": to_float_str(f1),
                "duration_sec_total": to_float_str(duration_sec_total),
                "false_alarms_per_hour": to_float_str(fa_per_hour),
                "mean_detection_delay_sec": to_float_str(mean_delay_sec) if mean_delay_sec is not None else "",
            }
        )

    comparison_rows = build_baseline_vs_proposed_rows(aggregate_rows)

    per_video_path = output_dir / "per_video_metrics.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    comparison_path = output_dir / "baseline_vs_proposed.csv"
    evaluation_meta_path = output_dir / "evaluation_metadata.json"

    write_csv(
        per_video_path,
        fieldnames=[
            "run_group",
            "config_name",
            "video_id",
            "split",
            "run_status",
            "eval_status",
            "tolerance_frames",
            "gt_events_count",
            "pred_events_count",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "input_fps",
            "duration_sec",
            "false_alarms_per_hour",
            "mean_detection_delay_sec",
            "output_dir",
            "error_message",
        ],
        rows=per_video_rows,
    )
    write_csv(
        aggregate_path,
        fieldnames=[
            "run_group",
            "config_name",
            "eval_status",
            "videos_total",
            "videos_evaluated",
            "videos_skipped",
            "gt_total",
            "pred_total",
            "tp_total",
            "fp_total",
            "fn_total",
            "precision",
            "recall",
            "f1",
            "duration_sec_total",
            "false_alarms_per_hour",
            "mean_detection_delay_sec",
        ],
        rows=aggregate_rows,
    )
    write_csv(
        comparison_path,
        fieldnames=["metric", "baseline", "proposed", "delta_proposed_minus_baseline"],
        rows=comparison_rows,
    )

    meta = {
        "generated_at_utc": utc_now_iso(),
        "run_group_dir": str(run_group_dir),
        "gt_dir": str(gt_dir),
        "tolerance_frames": args.tolerance_frames,
        "config_filter": sorted(cfg_filter),
        "split_filter": sorted(split_filter),
        "per_video_metrics_csv": str(per_video_path),
        "aggregate_metrics_csv": str(aggregate_path),
        "baseline_vs_proposed_csv": str(comparison_path),
    }
    evaluation_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Run group: {run_group_dir}")
    print(f"GT dir: {gt_dir}")
    print(f"Tolerance frames: {args.tolerance_frames}")
    print(f"Per-video metrics: {per_video_path}")
    print(f"Aggregate metrics: {aggregate_path}")
    print(f"Baseline vs proposed: {comparison_path}")
    print(f"Evaluation metadata: {evaluation_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
