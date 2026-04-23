"""
tracker.py — 하이브리드 추적 엔진
LK Optical Flow (속도) + ORB 앵커 검증 (정밀도) + Median Consensus (노이즈 제거)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Displacement:
    top_dx: float = 0.0   # 상단 구역 수평 변위 (pixel)
    top_dy: float = 0.0
    bot_dx: float = 0.0   # 하단 구역 수평 변위 (pixel)
    bot_dy: float = 0.0
    ref_dx: float = 0.0   # 고정 참조점 변위 (카메라 진동)
    ref_dy: float = 0.0
    top_pts_count: int = 0
    bot_pts_count: int = 0


class ZoneTracker:
    """단일 구역(ROI)의 LK + ORB 하이브리드 추적기"""

    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    FEATURE_PARAMS = dict(
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    REINIT_INTERVAL  = 90    # N 프레임마다 ORB 검증
    DRIFT_THRESHOLD  = 2.0   # 픽셀 드리프트 허용 한계
    MIN_POINTS       = 10    # 최소 추적 포인트 수

    def __init__(self, roi: tuple, label: str = "zone"):
        self.roi   = roi   # (x, y, w, h)
        self.label = label
        self.orb   = cv2.ORB_create(nfeatures=500)
        self.bf    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_gray:   Optional[np.ndarray] = None
        self.prev_pts:    Optional[np.ndarray] = None
        self.keyframe:    Optional[np.ndarray] = None
        self.kf_kp        = None
        self.kf_des       = None
        self.cumulative_dx = 0.0
        self.cumulative_dy = 0.0
        self.frame_count  = 0

    def _crop(self, gray: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi
        return gray[y:y+h, x:x+w]

    def _detect_points(self, gray_crop: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(gray_crop, mask=None, **self.FEATURE_PARAMS)
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return pts

    def _set_keyframe(self, gray_crop: np.ndarray):
        self.keyframe = gray_crop.copy()
        self.kf_kp, self.kf_des = self.orb.detectAndCompute(gray_crop, None)

    def initialize(self, gray: np.ndarray):
        crop = self._crop(gray)
        self.prev_gray = gray.copy()
        self.prev_pts  = self._detect_points(crop)
        self._set_keyframe(crop)
        self.cumulative_dx = 0.0
        self.cumulative_dy = 0.0
        self.frame_count   = 0

    def _orb_drift_check(self, gray_crop: np.ndarray) -> Optional[float]:
        """ORB로 키프레임 대비 드리프트 측정. 포인트 부족 시 None 반환."""
        if self.kf_des is None or len(self.kf_des) < 8:
            return None
        kp, des = self.orb.detectAndCompute(gray_crop, None)
        if des is None or len(des) < 8:
            return None
        matches = self.bf.match(self.kf_des, des)
        if len(matches) < 6:
            return None
        src = np.float32([self.kf_kp[m.queryIdx].pt for m in matches])
        dst = np.float32([kp[m.trainIdx].pt           for m in matches])
        delta = dst - src
        median_dx = float(np.median(delta[:, 0]))
        median_dy = float(np.median(delta[:, 1]))
        return float(np.hypot(median_dx - self.cumulative_dx,
                              median_dy - self.cumulative_dy))

    def update(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple[float, float, int]:
        """
        LK로 이번 프레임 변위 계산.
        Returns: (dx, dy, n_points)
        """
        crop = self._crop(gray)

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < self.MIN_POINTS:
            self.initialize(gray)
            return 0.0, 0.0, 0

        self.frame_count += 1

        # ── LK 추적 ──────────────────────────────────────────────────────
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.LK_PARAMS
        )

        # 마스크 적용 (동적 물체 영역 제외)
        if mask is not None:
            x, y, w, h = self.roi
            zone_mask = mask[y:y+h, x:x+w]
            for i, pt in enumerate(self.prev_pts):
                px, py = int(pt[0][0]), int(pt[0][1])
                px = min(max(px, 0), w-1)
                py = min(max(py, 0), h-1)
                if zone_mask[py, px] > 0:
                    status[i] = 0  # 동적 영역이면 무효화

        good_old = self.prev_pts[status.flatten() == 1]
        good_new = next_pts[status.flatten() == 1]      if next_pts is not None else np.empty((0, 1, 2))

        if len(good_new) < self.MIN_POINTS:
            self.initialize(gray)
            return 0.0, 0.0, 0

        # ── Median Consensus ─────────────────────────────────────────────
        deltas = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
        dx = float(np.median(deltas[:, 0]))
        dy = float(np.median(deltas[:, 1]))
        self.cumulative_dx += dx
        self.cumulative_dy += dy

        # ── ORB 드리프트 검증 (N프레임마다) ─────────────────────────────
        if self.frame_count % self.REINIT_INTERVAL == 0:
            drift = self._orb_drift_check(crop)
            if drift is not None and drift > self.DRIFT_THRESHOLD:
                print(f"[WARN] {self.label} 드리프트 {drift:.2f}px 감지 → 재초기화")
                self.initialize(gray)
                return 0.0, 0.0, 0

        # 포인트 수가 절반 이하로 줄면 새 포인트 보충
        if len(good_new) < self.FEATURE_PARAMS["maxCorners"] * 0.5:
            extra = self._detect_points(crop)
            if extra is not None and len(extra) > 0:
                good_new = np.concatenate([good_new.reshape(-1, 1, 2), extra], axis=0)

        self.prev_pts  = good_new.reshape(-1, 1, 2).astype(np.float32)
        self.prev_gray = gray.copy()

        return dx, dy, len(good_new)


class DynamicMask:
    """MOG2 배경 차분으로 움직이는 물체(사람, 중장비) 마스크 생성"""

    def __init__(self):
        self.mog = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=50, detectShadows=False
        )
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self._kopen  = kernel_open
        self._kclose = kernel_close

    def apply(self, frame: np.ndarray) -> np.ndarray:
        fg = self.mog.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._kopen)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kclose)
        return fg   # 255 = 동적 물체


class CLAHEPreprocessor:
    """국소 대비 강화 — 먼지·안개·강한 그림자 환경 대응"""

    def __init__(self, clip_limit=2.0, tile_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    def apply(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class HybridTracker:
    """전체 추적 파이프라인 통합 관리자"""

    def __init__(self, rois: list):
        """
        rois: [top_roi, bot_roi, ref_roi]  각 roi = (x, y, w, h)
        """
        self.top_tracker = ZoneTracker(rois[0], label="상단")
        self.bot_tracker = ZoneTracker(rois[1], label="하단")
        self.ref_tracker = ZoneTracker(rois[2], label="고정참조")
        self.rois        = rois

        self.preprocessor = CLAHEPreprocessor()
        self.dyn_mask     = DynamicMask()
        self._initialized = False

        # 화면 표시용 — 외부에서 접근
        self.debug_pts: dict = {"top": [], "bot": [], "ref": []}

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        enhanced = self.preprocessor.apply(frame)
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    def update(self, frame: np.ndarray) -> Displacement:
        gray = self._to_gray(frame)
        mask = self.dyn_mask.apply(frame)

        if not self._initialized:
            self.top_tracker.initialize(gray)
            self.bot_tracker.initialize(gray)
            self.ref_tracker.initialize(gray)
            self._initialized = True
            return Displacement()

        tdx, tdy, tn = self.top_tracker.update(gray, mask)
        bdx, bdy, bn = self.bot_tracker.update(gray, mask)
        rdx, rdy, _  = self.ref_tracker.update(gray, mask=None)  # 참조점은 마스킹 없음

        # 포인트 디버그 저장
        self.debug_pts["top"] = self.top_tracker.prev_pts
        self.debug_pts["bot"] = self.bot_tracker.prev_pts
        self.debug_pts["ref"] = self.ref_tracker.prev_pts

        return Displacement(
            top_dx=tdx, top_dy=tdy,
            bot_dx=bdx, bot_dy=bdy,
            ref_dx=rdx, ref_dy=rdy,
            top_pts_count=tn,
            bot_pts_count=bn,
        )

    def reinitialize(self, frame: np.ndarray):
        gray = self._to_gray(frame)
        self.top_tracker.initialize(gray)
        self.bot_tracker.initialize(gray)
        self.ref_tracker.initialize(gray)