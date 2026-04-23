"""
alerter.py  — 경보 판단
visualizer.py — 화면 오버레이
logger.py   — CSV 데이터 로깅
"""

# ═══════════════════════════════════════════════════════════════════
# alerter.py
# ═══════════════════════════════════════════════════════════════════

from dataclasses import dataclass
from converter import PhysicsResult
import time


@dataclass
class AlertState:
    active:   bool  = False
    level:    str   = "OK"      # "OK" | "WARN" | "DANGER"
    reason:   str   = ""
    tilt_cm:  float = 0.0
    tilt_deg: float = 0.0


class Alerter:
    """
    2단계 경보:
    WARN   = 임계값의 70% 초과
    DANGER = 임계값 초과
    """

    def __init__(self, alert_cm: float, alert_deg: float):
        self.alert_cm  = alert_cm
        self.alert_deg = alert_deg
        self._last_alert_time = 0.0
        self._alert_cooldown  = 5.0   # 초 (반복 경보 방지)

    def check(self, result: PhysicsResult) -> AlertState:
        cm  = result.tilt_total_cm
        deg = result.tilt_deg

        if cm >= self.alert_cm or deg >= self.alert_deg:
            level  = "DANGER"
            reason = f"기울기 {cm:.1f}cm / {deg:.2f}deg 임계값 초과"
            active = True
        elif cm >= self.alert_cm * 0.7 or deg >= self.alert_deg * 0.7:
            level  = "WARN"
            reason = f"기울기 {cm:.1f}cm / {deg:.2f}deg 주의 구간"
            active = False
        else:
            return AlertState(active=False, level="OK", tilt_cm=cm, tilt_deg=deg)

        now = time.time()
        if active and (now - self._last_alert_time) > self._alert_cooldown:
            self._last_alert_time = now
            print(f"[ALERT-{level}] {reason}")

        return AlertState(active=active, level=level, reason=reason, tilt_cm=cm, tilt_deg=deg)

