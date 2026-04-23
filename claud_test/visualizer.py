"""
visualizer.py — 화면 오버레이 렌더러
"""

import cv2
import numpy as np
from converter import PhysicsResult
from alerter import AlertState

_COLOR = {
    "OK":     (100, 220, 100),
    "WARN":   (0,   200, 255),
    "DANGER": (0,    60, 255),
}


class Visualizer:
    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    PANEL_W   = 360
    PANEL_H   = 230
    PANEL_PAD = 14

    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h

    def draw(self, frame: np.ndarray, tracker, result: PhysicsResult,
             alert: AlertState, fps: float) -> np.ndarray:
        out = frame.copy()

        zone_colors = {"top": (0, 255, 200), "bot": (255, 200, 0), "ref": (200, 200, 200)}
        roi_labels  = ["상단", "하단", "고정"]
        for i, (zone, color) in enumerate(zone_colors.items()):
            roi = tracker.rois[i]
            rx, ry, rw, rh = roi
            cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), color, 1)
            cv2.putText(out, roi_labels[i], (rx + 4, ry + 18),
                        self.FONT, 0.5, color, 1, cv2.LINE_AA)
            pts = tracker.debug_pts.get(zone)
            if pts is not None and len(pts) > 0:
                for pt in pts:
                    px = int(pt[0][0]) + rx
                    py = int(pt[0][1]) + ry
                    cv2.circle(out, (px, py), 2, color, -1)

        self._draw_panel(out, result, alert, fps)

        if alert.level == "DANGER":
            cv2.rectangle(out, (0, 0), (self.fw - 1, self.fh - 1), (0, 0, 255), 6)
        elif alert.level == "WARN":
            cv2.rectangle(out, (0, 0), (self.fw - 1, self.fh - 1), (0, 165, 255), 3)

        return out

    def _draw_panel(self, frame: np.ndarray, result: PhysicsResult,
                    alert: AlertState, fps: float):
        x0 = self.fw - self.PANEL_W - 10
        y0 = 10
        panel = np.zeros((self.PANEL_H, self.PANEL_W, 3), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (self.PANEL_W - 1, self.PANEL_H - 1), (25, 25, 25), -1)
        cv2.rectangle(panel, (0, 0), (self.PANEL_W - 1, self.PANEL_H - 1), (80, 80, 80), 1)

        color = _COLOR[alert.level]
        lines = [
            (f"상태: {alert.level}", color),
            (f"상단 변위:  X={result.top_dx_cm:+.2f}cm  Y={result.top_dy_cm:+.2f}cm", (220, 220, 220)),
            (f"하단 변위:  X={result.bot_dx_cm:+.2f}cm  Y={result.bot_dy_cm:+.2f}cm", (220, 220, 220)),
            (f"기울기(수평): {result.tilt_dx_cm:+.2f} cm", (220, 220, 220)),
            (f"기울기(각도): {result.tilt_deg:.3f} deg",   (220, 220, 220)),
            (f"합산 변위:   {result.tilt_total_cm:.2f} cm", (220, 220, 220)),
            (f"포인트: 상단 {result.top_pts}  하단 {result.bot_pts}", (160, 160, 160)),
            (f"FPS: {fps:.1f}" if fps > 0 else "FPS: --", (160, 160, 160)),
        ]
        for i, (line, c) in enumerate(lines):
            cv2.putText(panel, line, (self.PANEL_PAD, 26 + i * 26),
                        self.FONT, 0.46, c, 1, cv2.LINE_AA)

        roi_slice = frame[y0:y0 + self.PANEL_H, x0:x0 + self.PANEL_W]
        if roi_slice.shape[:2] == panel.shape[:2]:
            blended = cv2.addWeighted(roi_slice, 0.2, panel, 0.8, 0)
            frame[y0:y0 + self.PANEL_H, x0:x0 + self.PANEL_W] = blended