"""Применение EventConsolidatorV3 к существующему events.csv.

Полезно, если pipeline уже запускался ранее и events.csv лежит на диске,
а нужно получить consolidated_events.csv без перезапуска инференса.

Пример:
    python tools/consolidate_events.py --task hardhat \\
        --input runs/demo_hardhat/events.csv \\
        --output runs/demo_hardhat/consolidated_events.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from main import consolidate_events  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Применить EventConsolidatorV3 к raw events.csv."
    )
    parser.add_argument(
        "--task", choices=("hardhat", "vest"), required=True,
        help="Тип задачи (определяет набор параметров V3).",
    )
    parser.add_argument("--input", required=True, help="Путь к events.csv.")
    parser.add_argument(
        "--output", required=True,
        help="Куда писать consolidated_events.csv.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    n = consolidate_events(Path(args.input), Path(args.output), task=args.task)
    print(f"Записано событий после V3: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
