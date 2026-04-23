"""
converter.py — 픽셀 → 물리량 변환
삼각함수 기반 변위(cm) 및 기울기 각도(deg) 산출
"""

import math
import numpy as np
from dataclasses import dataclass
from compensator import CompensatedDisplacement


@dataclass
class PhysicsResult:
    # 절대 변위 (건물 전체 이동)
    top_dx_cm:   float = 0.0
    top_dy_cm:   float = 0.0
    bot_dx_cm:   float = 0.0
    bot_dy_cm:   float = 0.0
    # 기울기 (상단 - 하단 상대 변위)
    tilt_dx_cm:  float = 0.0
    tilt_dy_cm:  float = 0.0
    tilt_deg:    float = 0.0   # 수평 기울기 각도
    tilt_total_cm: float = 0.0 # 벡터 합산 변위 크기
    # 메타
    top_pts: int = 0
    bot_pts: int = 0


class PhysicsConverter:
    """
    [스케일 계산]
    실제 건물 높이(m) / 픽셀 높이(px) = cm_per_pixel
    건물이 화면에 수직으로 보이는 경우를 기준으로 함.

    [기울기 각도]
    tilt_deg = arctan(수평변위_cm / 건물높이_cm)
    """

    def __init__(self, cm_per_pixel: float, building_height_m: float):
        self.cmp  = cm_per_pixel          # cm / pixel
        self.bldg_height_cm = building_height_m * 100.0

    @staticmethod
    def calc_scale(building_height_m: float, pixel_height: int) -> float:
        """cm_per_pixel 계산"""
        if pixel_height <= 0:
            raise ValueError("pixel_height는 양수여야 합니다.")
        return (building_height_m * 100.0) / pixel_height

    def convert(self, comp: CompensatedDisplacement) -> PhysicsResult:
        top_dx_cm  = comp.top_dx * self.cmp
        top_dy_cm  = comp.top_dy * self.cmp
        bot_dx_cm  = comp.bot_dx * self.cmp
        bot_dy_cm  = comp.bot_dy * self.cmp
        tilt_dx_cm = comp.tilt_dx * self.cmp
        tilt_dy_cm = comp.tilt_dy * self.cmp

        tilt_total_cm = math.hypot(tilt_dx_cm, tilt_dy_cm)
        # 수평 기울기 각도 (arctan(수평변위 / 건물높이))
        tilt_deg = math.degrees(math.atan2(abs(tilt_dx_cm), self.bldg_height_cm))

        return PhysicsResult(
            top_dx_cm=top_dx_cm,  top_dy_cm=top_dy_cm,
            bot_dx_cm=bot_dx_cm,  bot_dy_cm=bot_dy_cm,
            tilt_dx_cm=tilt_dx_cm, tilt_dy_cm=tilt_dy_cm,
            tilt_deg=tilt_deg,
            tilt_total_cm=tilt_total_cm,
            top_pts=comp.top_pts_count,
            bot_pts=comp.bot_pts_count,
        )