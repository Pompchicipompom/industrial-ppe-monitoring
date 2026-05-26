from __future__ import annotations


def get_roi_rect(frame_w: int, frame_h: int, roi_cfg: dict) -> tuple[int, int, int, int]:
    x1 = int(frame_w * roi_cfg["x_ratio"])
    y1 = int(frame_h * roi_cfg["y_ratio"])
    x2 = int(frame_w * (roi_cfg["x_ratio"] + roi_cfg["w_ratio"]))
    y2 = int(frame_h * (roi_cfg["y_ratio"] + roi_cfg["h_ratio"]))
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))
    return x1, y1, x2, y2


def clip_xyxy_to_frame(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(x1), float(frame_w - 1)))
    y1 = max(0.0, min(float(y1), float(frame_h - 1)))
    x2 = max(0.0, min(float(x2), float(frame_w - 1)))
    y2 = max(0.0, min(float(y2), float(frame_h - 1)))
    if x2 <= x1:
        x2 = min(float(frame_w - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(frame_h - 1), y1 + 1.0)
    return x1, y1, x2, y2


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def box_center(box_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def clip_box_to_box(
    inner_box: tuple[float, float, float, float],
    outer_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    cx1 = max(ix1, ox1)
    cy1 = max(iy1, oy1)
    cx2 = min(ix2, ox2)
    cy2 = min(iy2, oy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def is_box_touching_roi_edge(
    box_xyxy: tuple[float, float, float, float],
    roi_rect: tuple[int, int, int, int],
    margin_px: int,
) -> bool:
    x1, y1, x2, y2 = box_xyxy
    rx1, ry1, rx2, ry2 = roi_rect
    m = max(0.0, float(margin_px))
    return x1 <= (rx1 + m) or y1 <= (ry1 + m) or x2 >= (rx2 - m) or y2 >= (ry2 - m)

