"""Командная утилита запуска модуля контроля СИЗ.

Запускает pipeline на одном видеофайле (или RTSP-источнике) и сохраняет:
- processed.mp4 — видео с наложенными bbox и статусами;
- events.csv / events.jsonl — сырые события из временной логики;
- consolidated_events.csv — события, объединённые EventConsolidatorV3;
- frame_metrics.csv — покадровая телеметрия;
- runtime_profile.json — тайминги стадий pipeline.

Пример:
    python main.py --config configs/production_hardhat.yaml \\
        --source input/demo.mp4 --output runs/demo_hardhat
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ppe_monitoring.config import load_config
from ppe_monitoring.event_consolidator import (
    ConsolidatorV3Params,
    EventConsolidatorV3,
    RawEvent,
)
from ppe_monitoring.pipeline import run_pipeline


# Параметры консолидатора, согласованные с финальной оценкой ВКР.
# Hardhat использует более узкие временные окна, vest — более широкие.
_CONSOLIDATOR_PRESETS: dict[str, dict] = {
    "hardhat": dict(
        same_episode_max_gap_seconds=6.0,
        expanded_anchor_bbox_ratio=3.0,
        center_distance_ratio=0.55,
        iou_threshold=0.10,
        history_seconds=6.0,
    ),
    "vest": dict(
        same_episode_max_gap_seconds=8.0,
        expanded_anchor_bbox_ratio=1.5,
        center_distance_ratio=0.75,
        iou_threshold=0.10,
        history_seconds=8.0,
    ),
}


def _output_overrides(output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "video_path": str(output_root / "processed.mp4"),
        "events_csv": str(output_root / "events.csv"),
        "events_jsonl": str(output_root / "events.jsonl"),
        "metrics_csv": str(output_root / "frame_metrics.csv"),
        "profile_json": str(output_root / "runtime_profile.json"),
    }


def _read_events_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("event_type")]


def _to_raw_events(rows: list[dict], video_id: str) -> list[RawEvent]:
    raw: list[RawEvent] = []
    for idx, r in enumerate(rows):
        bbox: tuple[float, float, float, float] | None = None
        try:
            bbox = (
                float(r.get("person_x1", 0) or 0),
                float(r.get("person_y1", 0) or 0),
                float(r.get("person_x2", 0) or 0),
                float(r.get("person_y2", 0) or 0),
            )
            if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                bbox = None
        except (TypeError, ValueError):
            bbox = None
        try:
            person_track_id: int | None = int(r["person_track_id"]) \
                if r.get("person_track_id") not in (None, "") else None
        except (TypeError, ValueError):
            person_track_id = None
        try:
            frame_idx = int(r.get("frame_idx", idx))
            timestamp_sec = float(r.get("timestamp_sec", 0.0))
        except (TypeError, ValueError):
            continue
        raw.append(RawEvent(
            video_id=str(r.get("video_id") or video_id),
            event_id=str(r.get("event_id", f"e_{idx + 1}")),
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            person_track_id=person_track_id,
            event_type=str(r.get("event_type", "no_violation")),
            bbox=bbox,
        ))
    return raw


def consolidate_events(events_csv: Path, output_csv: Path, task: str) -> int:
    """Применяет EventConsolidatorV3 к raw-событиям и пишет consolidated_events.csv.

    Возвращает количество событий после консолидации.
    """
    rows = _read_events_csv(events_csv)
    if not rows:
        output_csv.write_text(
            "video_id,event_id,frame_idx,timestamp_sec,person_track_id,"
            "event_type,person_x1,person_y1,person_x2,person_y2\n",
            encoding="utf-8",
        )
        return 0
    video_id = rows[0].get("video_id") or output_csv.parent.name
    raw_events = _to_raw_events(rows, video_id)
    params = ConsolidatorV3Params(**_CONSOLIDATOR_PRESETS[task])
    consolidator = EventConsolidatorV3(params)
    emitted, _ = consolidator.consolidate_video(raw_events)
    fields = [
        "video_id", "event_id", "frame_idx", "timestamp_sec",
        "person_track_id", "event_type",
        "person_x1", "person_y1", "person_x2", "person_y2",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in emitted:
            writer.writerow({
                "video_id": e.video_id,
                "event_id": e.event_id,
                "frame_idx": e.frame_idx,
                "timestamp_sec": f"{e.timestamp_sec:.3f}",
                "person_track_id": "" if e.person_track_id is None
                                   else e.person_track_id,
                "event_type": e.event_type,
                "person_x1": f"{e.bbox[0]:.2f}" if e.bbox else "",
                "person_y1": f"{e.bbox[1]:.2f}" if e.bbox else "",
                "person_x2": f"{e.bbox[2]:.2f}" if e.bbox else "",
                "person_y2": f"{e.bbox[3]:.2f}" if e.bbox else "",
            })
    return len(emitted)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Запуск pipeline событийного контроля СИЗ "
            "(детектор + tracker + temporal logic + EventConsolidatorV3)."
        ),
    )
    parser.add_argument(
        "--config", required=True,
        help="Путь к YAML-конфигу (например, configs/production_hardhat.yaml).",
    )
    parser.add_argument(
        "--source",
        help="Путь к видеофайлу или RTSP-источнику. "
             "Если не задан, используется значение из конфига.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Каталог, в который будут записаны processed.mp4, events.csv и т.д.",
    )
    parser.add_argument(
        "--task", choices=("hardhat", "vest", "auto"), default="auto",
        help="Тип задачи для пост-консолидации событий. "
             "По умолчанию определяется по имени конфига.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Ограничить количество обрабатываемых кадров (для отладки).",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Отключить превью-окно (для серверного запуска).",
    )
    return parser.parse_args(argv)


def _infer_task(args: argparse.Namespace) -> str:
    if args.task != "auto":
        return args.task
    name = Path(args.config).name.lower()
    if "vest" in name:
        return "vest"
    return "hardhat"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output).resolve()
    task = _infer_task(args)

    overrides: dict[str, dict] = {
        "pipeline": {},
        "output": _output_overrides(output_root),
    }
    if args.source is not None:
        overrides["pipeline"]["source"] = args.source
    if args.no_preview:
        overrides["pipeline"]["display_preview"] = False
    if args.max_frames is not None:
        overrides["pipeline"]["max_frames"] = int(args.max_frames)

    cfg = load_config(config_path=args.config, overrides=overrides)
    result = run_pipeline(cfg)

    events_csv = output_root / "events.csv"
    consolidated_csv = output_root / "consolidated_events.csv"
    n_raw = sum(1 for _ in _read_events_csv(events_csv))
    n_after = consolidate_events(events_csv, consolidated_csv, task=task)

    summary = {
        "task": task,
        "output_dir": str(output_root),
        "events_total": int(result.get("events_total", n_raw)),
        "raw_events_in_csv": n_raw,
        "events_after_v3": n_after,
    }
    print("\n=== Сводка по запуску ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
