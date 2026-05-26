"""Logical video_id derivation for events and experiment manifests (no OpenCV dependency)."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from urllib.parse import urlparse


def sanitize_video_id(raw: str) -> str:
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    value = "".join(out).strip("._")
    return value or "video_unknown"


def resolve_video_id(source_value, explicit_video_id: str | None) -> str:
    if explicit_video_id:
        cleaned = str(explicit_video_id).strip()
        if cleaned:
            return cleaned

    source_str = str(source_value).strip()
    if not source_str:
        return "video_unknown"
    if source_str.isdigit():
        return f"camera_{source_str}"

    lowered = source_str.lower()
    if lowered.startswith(("rtsp://", "rtsps://")):
        parsed = urlparse(source_str)
        host = parsed.hostname or "unknown_host"
        path_part = parsed.path.strip("/").replace("/", "_")
        if not path_part:
            path_part = "stream"
        return sanitize_video_id(f"rtsp_{host}_{path_part}")

    p = PureWindowsPath(source_str) if "\\" in source_str else Path(source_str)
    stem = p.stem if p.suffix else p.name
    return sanitize_video_id(stem)
