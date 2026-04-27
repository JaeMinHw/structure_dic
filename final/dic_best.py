"""

DIC (Digital Image Correlation) 실시간 모니터링 시스템

------------------------------------------------------

- 구조물의 상단/하단 추적점 변위 및 기울기 각도를 실시간 분석

- 기준점(고정 ROI)으로 카메라 흔들림 보정

- 서브픽셀 정밀도 적용

- 그래프 업데이트 주기 분리 (성능 최적화)

"""



import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # GUI 백엔드 명시 (충돌 방지)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import tkinter as tk
from tkinter import font as tkfont
from collections import deque


# ============================================================
# [CONFIG] 전역 설정
# ============================================================

# VIDEO_SOURCE    = 'C:/Users/admin_user/Desktop/dic_test2.mp4'
VIDEO_SOURCE = './destory_structure_video.mp4'


# UI
INIT_HEIGHT     = 5.0
MIN_HEIGHT      = 0.1
MAX_HEIGHT      = 100.0
STEP_HEIGHT     = 0.1
SCREEN_RATIO    = 0.88          # 창이 모니터를 차지하는 비율


# DIC
TEMPLATE_SIZE   = 150           # 템플릿 크기 (px)
SEARCH_MARGIN   = 200            # 탐색 마진 — 전체 이미지 탐색 대신 범위 제한
MARKER_SIZE     = 20
OVERLAY_ALPHA   = 0.4           # 고스트 투명도


# 데이터 / 그래프
HISTORY_LIMIT   = 300           # deque 최대 길이
GRAPH_UPDATE_INTERVAL = 10      # N 프레임마다 그래프 갱신
GRAPH_FIG_SIZE  = (10, 4)


# 매칭 신뢰도 임계값 (이 값 미만이면 해당 프레임 결과를 버림)
CONFIDENCE_THRESHOLD = 0.6
# ============================================================



# ──────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────
def get_screen_size() -> tuple[int, int]:
    """모니터 해상도 반환 (width, height)"""
    root = tk.Tk()
    root.withdraw()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return sw, sh



def calc_scale(img_w: int, img_h: int,
               sw: int, sh: int,
               ratio: float = SCREEN_RATIO) -> float:
    """이미지가 화면 비율 안에 들어오도록 스케일 계산 (최대 1.0)"""
    return min((sw * ratio) / img_w,
               (sh * ratio) / img_h,
               1.0)



def show_window_centered(win_name: str, w: int, h: int,
                          sw: int, sh: int) -> None:
    """OpenCV 창 생성 및 화면 중앙 배치"""
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, w, h)
    cv2.moveWindow(win_name,
                   (sw - w) // 2,
                   (sh - h) // 2)


def subpixel_peak(response: np.ndarray,
                   loc: tuple[int, int]) -> tuple[float, float]:
    """
    파라볼릭 보간으로 서브픽셀 정밀도 위치 계산.
    loc = (col, row) 형식.
    """
    px, py = loc
    rows, cols = response.shape
    if px < 1 or px >= cols - 1 or py < 1 or py >= rows - 1:
        return float(px), float(py)


    # X
    dx_n = response[py, px - 1] - response[py, px + 1]
    dx_d = 2.0 * (response[py, px - 1]
                  - 2 * response[py, px]
                  + response[py, px + 1])
    sx = px + (dx_n / dx_d if abs(dx_d) > 1e-10 else 0.0)


    # Y
    dy_n = response[py - 1, px] - response[py + 1, px]
    dy_d = 2.0 * (response[py - 1, px]
                  - 2 * response[py, px]
                  + response[py + 1, px])
    sy = py + (dy_n / dy_d if abs(dy_d) > 1e-10 else 0.0)


    return sx, sy





# ──────────────────────────────────────────────────────────────
# CLAHE 전처리
# ──────────────────────────────────────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def preprocess(img: np.ndarray) -> np.ndarray:
    """BGR → 그레이 + CLAHE"""
    return _clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))



# ──────────────────────────────────────────────────────────────
# 1. 건물 높이 입력 UI (Ruler Picker)
# ──────────────────────────────────────────────────────────────
def get_ruler_picker_ui(sw: int, sh: int) -> float:
    """
    슬라이더 + ＋/－ 버튼으로 건물 높이를 입력받는 tkinter UI.
    반환: 선택된 높이 (float, 단위 m)
    """
    result = {"height": INIT_HEIGHT}       # 클로저로 값 전달 (전역 변수 불필요)


    root = tk.Tk()
    root.title("구조물 사이즈 설정")
    win_w, win_h = 420, 580
    root.geometry(
        f"{win_w}x{win_h}+"
        f"{(sw - win_w) // 2}+{(sh - win_h) // 2}"
    )
    root.configure(bg="white")
    root.resizable(False, False)


    # 폰트
    title_f = tkfont.Font(family="맑은 고딕", size=15, weight="bold")
    value_f = tkfont.Font(family="Arial",    size=40, weight="bold")
    sub_f   = tkfont.Font(family="Arial",    size=16)
    unit_f  = tkfont.Font(family="Arial",    size=14, weight="bold")


    tk.Label(root, text="구조물 높이를 입력해주세요",
             bg="white", font=title_f, pady=30).pack()


    # 슬라이더 좌우 숫자 레이블
    label_frame = tk.Frame(root, bg="white")
    label_frame.pack(pady=4)
    labels = []
    for i in range(5):
        fg = "#333333" if i == 2 else "#D0D0D0"
        fnt = value_f if i == 2 else sub_f
        lbl = tk.Label(label_frame, text="", bg="white", fg=fg, font=fnt)
        lbl.pack(side="left", padx=10)
        labels.append(lbl)


    tk.Label(root, text="meters", font=unit_f,
             bg="white", fg="gray").pack()

    def update_ruler(val: str) -> None:
        """슬라이더 이동 시 숫자 레이블 갱신"""
        v = float(val)
        result["height"] = v
        offsets = [-STEP_HEIGHT * 2, -STEP_HEIGHT, 0,
                    STEP_HEIGHT,      STEP_HEIGHT * 2]
        for i, off in enumerate(offsets):
            dv = v + off
            labels[i].config(text=f"{dv:.1f}" if dv >= 0 else "")


    # 슬라이더
    style_frame = tk.Frame(root, bg="white", pady=30)
    style_frame.pack(fill="x", padx=40)
    scale_w = tk.Scale(
        style_frame,
        from_=MIN_HEIGHT, to=MAX_HEIGHT,
        orient="horizontal",
        resolution=STEP_HEIGHT,
        showvalue=False,
        bg="white", bd=0,
        highlightthickness=0,
        troughcolor="#F0F0F0",
        activebackground="#4CAF50",
        length=340,
        command=update_ruler
    )
    scale_w.set(INIT_HEIGHT)
    scale_w.pack()


    # ＋ / － 버튼
    btn_f = tk.Frame(root, bg="white")
    btn_f.pack(pady=10)


    def adjust(amount: float) -> None:
        new_val = round(scale_w.get() + amount, 1)
        if MIN_HEIGHT <= new_val <= MAX_HEIGHT:
            scale_w.set(new_val)


    for txt, amt in [("－", -STEP_HEIGHT), ("＋", STEP_HEIGHT)]:
        tk.Button(
            btn_f, text=txt, font=("Arial", 14), width=4,
            command=lambda a=amt: adjust(a),
            bg="#F8F8F8", bd=1
        ).pack(side="left", padx=20)


    # 저장 버튼
    def on_confirm() -> None:
        result["height"] = scale_w.get()
        root.destroy()


    tk.Button(
        root, text="저장 및 분석 시작",
        font=("맑은 고딕", 12, "bold"),
        bg="#4CAF50", fg="white",
        width=25, height=2, bd=0,
        command=on_confirm, cursor="hand2"
    ).pack(side="bottom", pady=40)


    # 초기 레이블 렌더링
    update_ruler(str(INIT_HEIGHT))
    root.mainloop()
    return result["height"]



# ──────────────────────────────────────────────────────────────
# 2. 마우스 ROI 선택
# ──────────────────────────────────────────────────────────────

class ROISelector:
    """
    클래스로 캡슐화하여 전역 변수 없이 마우스 ROI를 선택.
    """
    def __init__(self) -> None:
        self._start: tuple[int, int] | None = None
        self._end:   tuple[int, int] | None = None
        self._selecting = False
        self._base_frame: np.ndarray | None = None
        self._temp_frame: np.ndarray | None = None
        self._done = False


    def _callback(self, event: int, x: int, y: int,
                  flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._start     = (x, y)
            self._end       = None
            self._selecting = True
            self._done      = False

        elif event == cv2.EVENT_MOUSEMOVE and self._selecting:
            if self._base_frame is not None:
                self._temp_frame = self._base_frame.copy()
                cv2.rectangle(self._temp_frame,
                              self._start, (x, y),
                              (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP and self._selecting:
            self._end       = (x, y)
            self._selecting = False
            self._done      = True


    def select(self, win_name: str,
               frame: np.ndarray,
               prompt: str = "") -> tuple[int, int, int, int] | None:
        """
        드래그로 ROI를 선택.
        반환: (x, y, w, h) — 실패 시 None
        """
        self._base_frame = frame.copy()
        self._temp_frame = frame.copy()
        self._done       = False


        cv2.setMouseCallback(win_name, self._callback)
        if prompt:
            print(f"[ACTION] {prompt}")


        while True:
            cv2.imshow(win_name, self._temp_frame)
            key = cv2.waitKey(16) & 0xFF


            if key == ord('q'):
                cv2.setMouseCallback(win_name, lambda *a: None)
                return None


            if self._done and self._start and self._end:
                cv2.setMouseCallback(win_name, lambda *a: None)
                x1, y1 = self._start
                x2, y2 = self._end
                return (min(x1, x2), min(y1, y2),
                        abs(x1 - x2), abs(y1 - y2))



# ──────────────────────────────────────────────────────────────
# 3. DIC 추적기
# ──────────────────────────────────────────────────────────────
class DICTracker:
    def __init__(self, gray_init: np.ndarray, center_x: int, center_y: int, 
                 template_size: int = TEMPLATE_SIZE, 
                 search_margin: int = SEARCH_MARGIN,
                 update_enabled: bool = True):
        
        self.tpl_size = template_size
        self.margin = search_margin
        self.update_enabled = update_enabled
        
        H, W = gray_init.shape
        half = template_size // 2
        tx1 = max(0, center_x - half); ty1 = max(0, center_y - half)
        tx2 = min(W, tx1 + template_size); ty2 = min(H, ty1 + template_size)
        
        self.template = gray_init[ty1:ty2, tx1:tx2].copy()
        self.th, self.tw = self.template.shape
        
        # 초기 위치 설정
        self.init_cx = float(tx1 + self.tw / 2.0)
        self.init_cy = float(ty1 + self.th / 2.0)
        
        self.cur_cx = self.init_cx
        self.cur_cy = self.init_cy

        # --- [에러 해결 핵심: 변수 초기화] ---
        self.prev_cx = self.init_cx
        self.prev_cy = self.init_cy
        # ----------------------------------

        self._H, self._W = H, W
        self.pos_history_x = deque(maxlen=5)
        self.pos_history_y = deque(maxlen=5)
        self.UPDATE_THRESHOLD = 0.90

    def track(self, gray: np.ndarray) -> tuple[float, float, float]:
        half = self.tpl_size // 2
        sx1 = max(0, int(self.cur_cx) - half - self.margin)
        sy1 = max(0, int(self.cur_cy) - half - self.margin)
        sx2 = min(self._W, int(self.cur_cx) + half + self.margin)
        sy2 = min(self._H, int(self.cur_cy) + half + self.margin)

        search = gray[sy1:sy2, sx1:sx2]
        if (search.shape[0] < self.th or search.shape[1] < self.tw):
            # 영역 이탈 시에도 관성 적용 시도
            dx = self.cur_cx - self.prev_cx
            dy = self.cur_cy - self.prev_cy
            self.prev_cx, self.prev_cy = self.cur_cx, self.cur_cy # 현재를 이전으로 저장
            self.cur_cx += dx
            self.cur_cy += dy
            return self.cur_cx, self.cur_cy, 0.0

        resp = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, conf, _, loc = cv2.minMaxLoc(resp)

        if conf >= CONFIDENCE_THRESHOLD:
            sub_x, sub_y = subpixel_peak(resp, loc)
            raw_cx = sx1 + sub_x + self.tw / 2.0
            raw_cy = sy1 + sub_y + self.th / 2.0

            # 이동 평균 필터 적용
            self.pos_history_x.append(raw_cx)
            self.pos_history_y.append(raw_cy)
            
            # --- [이전 위치 업데이트 전 저장] ---
            self.prev_cx = self.cur_cx
            self.prev_cy = self.cur_cy
            # ----------------------------------

            self.cur_cx = sum(self.pos_history_x) / len(self.pos_history_x)
            self.cur_cy = sum(self.pos_history_y) / len(self.pos_history_y)

            if self.update_enabled and conf > self.UPDATE_THRESHOLD:
                nx1 = int(self.cur_cx - self.tw / 2.0)
                ny1 = int(self.cur_cy - self.th / 2.0)
                if nx1 >= 0 and ny1 >= 0 and nx1 + self.tw <= self._W and ny1 + self.th <= self._H:
                    new_tpl = gray[ny1:ny1+self.th, nx1:nx1+self.tw].copy()
                    self.template = cv2.addWeighted(self.template, 0.7, new_tpl, 0.3, 0)
        else:
            # --- [신뢰도 낮을 때 관성 추적 로직] ---
            dx = self.cur_cx - self.prev_cx
            dy = self.cur_cy - self.prev_cy
            
            self.prev_cx = self.cur_cx # 현재 위치를 백업
            self.prev_cy = self.cur_cy
            
            self.cur_cx += dx # 직전 이동량만큼 강제로 이동
            self.cur_cy += dy
            # ------------------------------------
        
        return self.cur_cx, self.cur_cy, float(conf)


# ──────────────────────────────────────────────────────────────
# 4. 실시간 그래프 관리
# ──────────────────────────────────────────────────────────────
class LivePlot:
    """
    matplotlib 실시간 그래프 (변위 cm + 각도 deg 이중 Y축).
    update() 호출 시 데이터만 추가하고,
    render() 호출 시 실제로 그림을 갱신해 CPU 부하를 분리.
    """

    def __init__(self, building_height_m: float) -> None:
        self.times: deque[datetime] = deque(maxlen=HISTORY_LIMIT)
        self.cms:   deque[float]    = deque(maxlen=HISTORY_LIMIT)
        self.degs:  deque[float]    = deque(maxlen=HISTORY_LIMIT)


        plt.ion()
        self.fig, ax1 = plt.subplots(figsize=GRAPH_FIG_SIZE)
        self.fig.patch.set_facecolor("#F9F9F9")


        # 변위 축 (좌)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Displacement (cm)", color="tab:blue", fontweight="bold")
        self.line_cm, = ax1.plot([], [], color="tab:blue",
                                  linewidth=1.5, label="Disp (cm)")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax1.grid(True, linestyle="--", alpha=0.4)


        # 각도 축 (우)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Angle (deg)", color="tab:red", fontweight="bold")
        self.line_deg, = ax2.plot([], [], color="tab:red",
                                   linewidth=1.5, linestyle="--",
                                   label="Angle (deg)")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        ax1.set_title(
            f"DIC Real-time Monitoring  "
            f"(Building Height: {building_height_m:.1f} m)",
            fontweight="bold"
        )
        self.fig.autofmt_xdate()
        self._ax1, self._ax2 = ax1, ax2


        # 범례 통합
        lines  = [self.line_cm, self.line_deg]
        labels = ["Disp (cm)", "Angle (deg)"]
        ax1.legend(lines, labels, loc="upper left", fontsize=9)


    def push(self, t: datetime, cm: float, deg: float) -> None:
        """데이터 추가 (매 프레임 호출 가능)"""
        self.times.append(t)
        self.cms.append(cm)
        self.degs.append(deg)


    def render(self) -> None:
        """그래프 실제 갱신 (N 프레임마다 호출)"""
        t_list = list(self.times)
        self.line_cm.set_data(t_list, list(self.cms))
        self.line_deg.set_data(t_list, list(self.degs))
        self._ax1.relim(); self._ax1.autoscale_view()
        self._ax2.relim(); self._ax2.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ──────────────────────────────────────────────────────────────
# 5. 초기화 및 추적점 설정 헬퍼
# ──────────────────────────────────────────────────────────────
def find_best_feature(gray: np.ndarray,
                       roi: tuple[int, int, int, int],
                       fallback_center: bool = True
                       ) -> tuple[int, int]:
    """
    ROI 내 최적 특징점 1개 탐색.
    실패 시 ROI 중심 반환 (fallback_center=True).
    """
    x, y, w, h = roi
    region = gray[y:y + h, x:x + w]
    pts = cv2.goodFeaturesToTrack(
        region, maxCorners=1,
        qualityLevel=0.01, minDistance=10
    )
    if pts is not None:
        fx, fy = pts[0][0]
        return int(x + fx), int(y + fy)
    if fallback_center:
        print(f"[WARN] ROI({x},{y},{w},{h}) 에서 특징점을 찾지 못해 중심점을 사용합니다.")
        return x + w // 2, y + h // 2

    raise RuntimeError(f"특징점 탐색 실패: ROI={roi}")



# ──────────────────────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────────────────────
def main() -> None:

    # ── 영상 열기 ───────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {VIDEO_SOURCE}")
        sys.exit(1)


    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] 첫 프레임을 읽을 수 없습니다.")
        sys.exit(1)


    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0          # FPS 정보가 없으면 30으로 가정

    wait_ms = max(1, int(1000 / video_fps))
    print(f"[INFO] 영상 FPS: {video_fps:.2f}  →  프레임 대기: {wait_ms} ms")


    img_h, img_w = first_frame.shape[:2]
    sw, sh       = get_screen_size()
    scale        = calc_scale(img_w, img_h, sw, sh)
    disp_w       = int(img_w * scale)
    disp_h       = int(img_h * scale)


    print(f"[INFO] 영상 해상도: {img_w}×{img_h}  →  표시: {disp_w}×{disp_h}")


    # ── ROI 선택 창 ─────────────────────────────────────────
    WIN_SETUP = "DIC Setup — select ROI"
    show_window_centered(WIN_SETUP, disp_w, disp_h, sw, sh)

    # 표시용 리사이즈 (선택은 표시 해상도에서 수행)
    disp_first = cv2.resize(first_frame, (disp_w, disp_h),
                             interpolation=cv2.INTER_LANCZOS4)


    selector = ROISelector()


    roi_building_disp = selector.select(
        WIN_SETUP, disp_first,
        prompt="[1/2] 건물(분석 대상) 영역을 드래그하세요. (취소: q)"
    )
    if roi_building_disp is None:
        print("[EXIT] 건물 ROI 선택 취소."); cv2.destroyAllWindows(); sys.exit(0)


    # 선택 영역 표시 후 다음 ROI 선택
    preview = disp_first.copy()
    x, y, w, h = roi_building_disp
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(preview, "Building ROI", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(WIN_SETUP, preview)
    cv2.waitKey(500)


    roi_fixed_disp = selector.select(
        WIN_SETUP, preview,
        prompt="[2/2] 고정 기준점(카메라 흔들림 보정용) 영역을 드래그하세요. (취소: q)"
    )
    if roi_fixed_disp is None:
        print("[EXIT] 고정점 ROI 선택 취소."); cv2.destroyAllWindows(); sys.exit(0)


    cv2.destroyWindow(WIN_SETUP)


    # ── 화면→원본 좌표 변환 ─────────────────────────────────
    def to_orig(roi_d: tuple) -> tuple[int, int, int, int]:
        return tuple(int(v / scale) for v in roi_d)


    roi_building = to_orig(roi_building_disp)
    roi_fixed    = to_orig(roi_fixed_disp)


    # ── 건물 높이 입력 ───────────────────────────────────────
    building_h_m  = get_ruler_picker_ui(sw, sh)
    building_h_cm = building_h_m * 100.0
    print(f"[INFO] 건물 높이: {building_h_m:.2f} m ({building_h_cm:.1f} cm)")


    # ── 전처리 & 특징점 추출 ─────────────────────────────────
    gray_init = preprocess(first_frame)


    # 건물 상단 15% / 하단 15% 영역에서 특징점 탐색
    bx, by, bw, bh = roi_building
    top_roi = (bx, by,              bw, max(1, int(bh * 0.15)))
    bot_roi = (bx, by + int(bh * 0.85), bw, max(1, int(bh * 0.15)))


    top_pt = find_best_feature(gray_init, top_roi)
    bot_pt = find_best_feature(gray_init, bot_roi)
    ref_pt = find_best_feature(gray_init, roi_fixed)


    print(f"[INFO] 상단 추적점: {top_pt}")
    print(f"[INFO] 하단 추적점: {bot_pt}")
    print(f"[INFO] 기준  추적점: {ref_pt}")


    # ── DIC 추적기 초기화 ────────────────────────────────────
    tracker_top = DICTracker(gray_init, *top_pt)
    tracker_bot = DICTracker(gray_init, *bot_pt)
    tracker_ref = DICTracker(gray_init, *ref_pt)


    # 픽셀 → 실측 변환 기준
    # 상단~하단 픽셀 거리 : 실제 높이 = pixel_dist : building_h_cm
    pixel_dist = max(1.0, abs(tracker_top.init_cy - tracker_bot.init_cy))
    px_per_cm  = pixel_dist / building_h_cm   # [px / cm]
    print(f"[INFO] 수직 픽셀 거리: {pixel_dist:.1f} px  →  "f"1 cm = {px_per_cm:.3f} px")


    # ── 그래프 초기화 ────────────────────────────────────────
    live_plot = LivePlot(building_h_m)


    # ── 분석 창 ──────────────────────────────────────────────
    WIN_ANALYSIS = "DIC Analysis Monitor  (q: 종료)"
    show_window_centered(WIN_ANALYSIS, disp_w, disp_h, sw, sh)


    frame_idx   = 0
    low_conf_cnt = 0


    # ── 메인 루프 ────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] 영상 종료.")
            break


        gray = preprocess(frame)


        # DIC 추적
        top_cx, top_cy, conf_t = tracker_top.track(gray)
        bot_cx, bot_cy, conf_b = tracker_bot.track(gray)
        ref_cx, ref_cy, conf_r = tracker_ref.track(gray)
        collapse_flag = False
        consecutive_low_conf = 0

        # 신뢰도 경고 (낮은 신뢰도 누적 시 알림)
        if min(conf_t, conf_b) < CONFIDENCE_THRESHOLD:
            consecutive_low_conf += 1
            # 10프레임 이상 추적 실패 시 붕괴로 판단
            if consecutive_low_conf > 10:
                collapse_flag = True
        else:
            consecutive_low_conf = 0

        # 시각화 부분
        if collapse_flag:
            cv2.putText(vis, "!!! COLLAPSE DETECTED !!!", (disp_w//4, disp_h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

        # ── 변위 계산 ─────────────────────────────────────────
        # 카메라 흔들림 보정: 기준점(ref) 이동량을 빼줌
        dx_ref = ref_cx - tracker_ref.init_cx
        dy_ref = ref_cy - tracker_ref.init_cy


        # 보정된 상단·하단 X 이동
        top_dx_corr = (top_cx - tracker_top.init_cx) - dx_ref
        bot_dx_corr = (bot_cx - tracker_bot.init_cx) - dx_ref


        # 상단과 하단의 X 방향 상대 변위 차 (기울기 원인)
        relative_dx = top_dx_corr - bot_dx_corr   # [px]


        # cm 변환
        disp_cm  = relative_dx / px_per_cm


        # 각도 계산 (pixel_dist = 수직 거리 기준)
        angle_deg = np.degrees(np.arctan2(relative_dx, pixel_dist))


        now = datetime.now()
        live_plot.push(now, disp_cm, angle_deg)


        # ── 그래프 갱신 (N 프레임마다) ───────────────────────
        if frame_idx % GRAPH_UPDATE_INTERVAL == 0:
            live_plot.render()


        # ── 시각화 프레임 구성 ───────────────────────────────
        vis = frame.copy()

        # 건물 ROI 고스트 오버레이
        bx, by, bw, bh = roi_building

        roi_blend = cv2.addWeighted(
            frame[by:by + bh, bx:bx + bw], 1 - OVERLAY_ALPHA,
            first_frame[by:by + bh, bx:bx + bw], OVERLAY_ALPHA,0
        )

        vis[by:by + bh, bx:bx + bw] = roi_blend

        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh),
                      (200, 200, 200), 1)

        # 추적 마커
        for pt, color, label in [
            ((int(top_cx), int(top_cy)), (255, 80,  0),  "TOP"),
            ((int(bot_cx), int(bot_cy)), (0,  200, 80),  "BOT"),
            ((int(ref_cx), int(ref_cy)), (0,   80, 255), "REF"),
        ]:

            cv2.drawMarker(vis, pt, color,cv2.MARKER_CROSS, MARKER_SIZE, 2)

            cv2.putText(vis, label,
                        (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1)

        # 상단↔하단 연결선 (변위 시각화)
        cv2.line(vis,
                 (int(top_cx), int(top_cy)),
                 (int(bot_cx), int(bot_cy)),
                 (200, 200, 0), 1)



        # HUD 텍스트
        hud_lines = [
            f"Disp : {disp_cm:+.3f} cm",
            f"Angle: {angle_deg:+.4f} deg",
            f"Conf : T={conf_t:.2f} B={conf_b:.2f} R={conf_r:.2f}",
            f"Frame: {frame_idx}",
        ]

        for i, txt in enumerate(hud_lines):
            cv2.putText(vis, txt, (18, 40 + i * 32),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), 2)

        # 표시용 리사이즈
        vis_disp = cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(WIN_ANALYSIS, vis_disp)
        frame_idx += 1

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            print("[INFO] 사용자 종료.")
            break



    # ── 정리 ────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()
    print("[INFO] 분석 완료.")



if __name__ == "__main__":
    main()