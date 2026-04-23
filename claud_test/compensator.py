"""
compensator.py — 카메라 진동 보정 모듈
Fixed Ref 차감 + Kalman 필터 (고주파 진동 제거)
"""

import cv2
import numpy as np
from tracker import Displacement
from dataclasses import dataclass


@dataclass
class CompensatedDisplacement:
    top_dx: float = 0.0   # 보정된 상단 수평 변위 (pixel)
    top_dy: float = 0.0
    bot_dx: float = 0.0   # 보정된 하단 수평 변위 (pixel)
    bot_dy: float = 0.0
    tilt_dx: float = 0.0  # 상단 - 하단 (기울기 성분, pixel)
    tilt_dy: float = 0.0
    top_pts_count: int = 0
    bot_pts_count: int = 0


def _make_kalman(process_noise=1e-4, measurement_noise=1e-2) -> cv2.KalmanFilter:
    """
    2D 위치 Kalman 필터 (상태: [x, y, vx, vy], 측정: [x, y])
    건물 이동은 매우 느리므로 process_noise를 작게 설정.
    """
    kf = cv2.KalmanFilter(4, 2)
    dt = 1.0  # 프레임당 시간 단위

    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    kf.processNoiseCov      = np.eye(4, dtype=np.float32) * process_noise
    kf.measurementNoiseCov  = np.eye(2, dtype=np.float32) * measurement_noise
    kf.errorCovPost         = np.eye(4, dtype=np.float32)
    kf.statePost            = np.zeros((4, 1), dtype=np.float32)
    return kf


class MotionCompensator:
    """
    Fixed Ref 차감 → Kalman 스무딩 2단계 파이프라인.

    [원리]
    - 카메라가 흔들리면 상단·하단·참조점 모두 동일하게 이동.
    - 참조점 이동량을 빼면 카메라 진동 성분이 제거됨.
    - 남은 잔류 노이즈는 Kalman 필터로 스무딩.
    """

    # 참조점 이동량의 이상치 감지 임계값 (pixel)
    # 이 이상이면 카메라 jump (충격) 으로 간주, 그 프레임은 보정값 0 사용
    REF_JUMP_THRESHOLD = 30.0

    def __init__(self):
        self.kf_top = _make_kalman()
        self.kf_bot = _make_kalman()

        # 누적 참조점 위치 (점진적 드리프트 추적용)
        self._ref_cum_x = 0.0
        self._ref_cum_y = 0.0

    def reset(self):
        self.kf_top = _make_kalman()
        self.kf_bot = _make_kalman()
        self._ref_cum_x = 0.0
        self._ref_cum_y = 0.0

    def _kalman_update(self, kf: cv2.KalmanFilter, dx: float, dy: float) -> tuple[float, float]:
        kf.predict()
        measurement = np.array([[dx], [dy]], dtype=np.float32)
        corrected   = kf.correct(measurement)
        return float(corrected[0, 0]), float(corrected[1, 0])

    def compensate(self, raw: Displacement) -> CompensatedDisplacement:
        ref_dx, ref_dy = raw.ref_dx, raw.ref_dy

        # 참조점 점프 감지 (충격·진동으로 순간 튀는 경우)
        if abs(ref_dx) > self.REF_JUMP_THRESHOLD or abs(ref_dy) > self.REF_JUMP_THRESHOLD:
            ref_dx, ref_dy = 0.0, 0.0  # 해당 프레임 보정 무시

        # ── Fixed Ref 차감 ────────────────────────────────────────────────
        comp_top_dx = raw.top_dx - ref_dx
        comp_top_dy = raw.top_dy - ref_dy
        comp_bot_dx = raw.bot_dx - ref_dx
        comp_bot_dy = raw.bot_dy - ref_dy

        # ── Kalman 스무딩 ─────────────────────────────────────────────────
        smooth_top_dx, smooth_top_dy = self._kalman_update(self.kf_top, comp_top_dx, comp_top_dy)
        smooth_bot_dx, smooth_bot_dy = self._kalman_update(self.kf_bot, comp_bot_dx, comp_bot_dy)

        return CompensatedDisplacement(
            top_dx=smooth_top_dx,
            top_dy=smooth_top_dy,
            bot_dx=smooth_bot_dx,
            bot_dy=smooth_bot_dy,
            tilt_dx=smooth_top_dx - smooth_bot_dx,
            tilt_dy=smooth_top_dy - smooth_bot_dy,
            top_pts_count=raw.top_pts_count,
            bot_pts_count=raw.bot_pts_count,
        )