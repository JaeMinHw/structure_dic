"""
logger.py — CSV 데이터 로거
"""

import csv
import os
from datetime import datetime
from converter import PhysicsResult
from alerter import AlertState


class DataLogger:
    FIELDS = [
        "timestamp", "frame",
        "top_dx_cm", "top_dy_cm", "bot_dx_cm", "bot_dy_cm",
        "tilt_dx_cm", "tilt_dy_cm", "tilt_deg", "tilt_total_cm",
        "top_pts", "bot_pts", "alert_level",
    ]

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"monitor_{ts}.csv")
        self._file   = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._buffer     = []
        self._flush_every = 30
        print(f"[LOG] 기록 시작: {path}")

    def log(self, frame_idx: int, result: PhysicsResult, alert: AlertState):
        row = {
            "timestamp":     datetime.now().isoformat(timespec="milliseconds"),
            "frame":         frame_idx,
            "top_dx_cm":     round(result.top_dx_cm,    3),
            "top_dy_cm":     round(result.top_dy_cm,    3),
            "bot_dx_cm":     round(result.bot_dx_cm,    3),
            "bot_dy_cm":     round(result.bot_dy_cm,    3),
            "tilt_dx_cm":    round(result.tilt_dx_cm,   3),
            "tilt_dy_cm":    round(result.tilt_dy_cm,   3),
            "tilt_deg":      round(result.tilt_deg,      4),
            "tilt_total_cm": round(result.tilt_total_cm, 3),
            "top_pts":       result.top_pts,
            "bot_pts":       result.bot_pts,
            "alert_level":   alert.level,
        }
        self._buffer.append(row)
        if len(self._buffer) >= self._flush_every:
            self._writer.writerows(self._buffer)
            self._file.flush()
            self._buffer.clear()

    def close(self):
        if self._buffer:
            self._writer.writerows(self._buffer)
        self._file.close()
        print("[LOG] 기록 종료.")