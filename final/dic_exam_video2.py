import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import tkinter as tk
from tkinter import font as tkfont
from collections import deque

# ==========================================
# [CONFIG] 모든 설정값 관리
# ==========================================
VIDEO_SOURCE = './destory_structure_video.mp4'

# 1. UI 및 높이 설정 관련
INIT_HEIGHT = 5.0      
MIN_HEIGHT = 0.1       
MAX_HEIGHT = 100.0     
STEP_HEIGHT = 0.1      

# 2. DIC 및 시각화 관련
TEMPLATE_SIZE = 120    # [수정] 100 -> 120 (안정성 강화)
SEARCH_MARGIN = 120    # [추가] 이전 위치 기준 탐색 범위 (특징점 튐 방지)
CONFIDENCE_THRESHOLD = 0.55 # [추가] 매칭 최소 신뢰도
UPDATE_THRESHOLD = 0.90     # [추가] 템플릿 갱신 기준 (매우 확실할 때만)
MARKER_SIZE = 20       
OVERLAY_ALPHA = 0.4    

# 3. 그래프 및 데이터 관련
HISTORY_LIMIT = 200    
GRAPH_FIG_SIZE = (10, 5) 

# ==========================================
# 유틸리티 함수 (서브픽셀 및 로컬 추적)
# ==========================================

def get_subpixel_peak(res, loc):
    """파라볼릭 보간으로 소수점 단위 좌표 계산"""
    x, y = loc
    h, w = res.shape
    if 1 < x < w - 1 and 1 < y < h - 1:
        dx_n = res[y, x+1] - res[y, x-1]
        dx_d = 2 * (2 * res[y, x] - res[y, x+1] - res[y, x-1])
        dy_n = res[y+1, x] - res[y-1, x]
        dy_d = 2 * (2 * res[y, x] - res[y+1, x] - res[y-1, x])
        sx = x + (dx_n / dx_d if abs(dx_d) > 1e-5 else 0)
        sy = y + (dy_n / dy_d if abs(dy_d) > 1e-5 else 0)
        return sx, sy
    return float(x), float(y)

def track_localized(img, tpl, prev_pos, margin):
    """이전 위치 주변(margin)에서만 템플릿을 찾아 특징점 튐 방지"""
    px, py = int(prev_pos[0]), int(prev_pos[1])
    th, tw = tpl.shape[:2]
    
    x1, y1 = max(0, px - tw//2 - margin), max(0, py - th//2 - margin)
    x2, y2 = min(img.shape[1], px + tw//2 + margin), min(img.shape[0], py + th//2 + margin)
    
    search_area = img[y1:y2, x1:x2]
    if search_area.shape[0] < th or search_area.shape[1] < tw:
        return prev_pos, 0.0, (0, 0) # 영역 이탈

    res = cv2.matchTemplate(search_area, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    sx, sy = get_subpixel_peak(res, max_loc)
    curr_pos = (x1 + sx + tw/2, y1 + sy + th/2)
    
    return curr_pos, max_val, (tw, th)

# ==========================================
# 1. 룰러(Ruler) 스타일 마우스 피커 UI
# ==========================================
final_height = INIT_HEIGHT
def get_ruler_picker_ui(screen_w, screen_h):
    global final_height
    root = tk.Tk()
    root.title("구조물 사이즈 설정")
    win_w, win_h = 400, 580
    root.geometry(f"{win_w}x{win_h}+{int(screen_w/2 - win_w/2)}+{int(screen_h/2 - win_h/2)}")
    root.configure(bg="white")
    
    title_f = tkfont.Font(family="맑은 고딕", size=15, weight="bold")
    value_f = tkfont.Font(family="Arial", size=40, weight="bold")
    sub_f = tkfont.Font(family="Arial", size=16) 
    unit_f = tkfont.Font(family="Arial", size=14, weight="bold")

    tk.Label(root, text="구조물 사이즈를 알려주세요", bg="white", font=title_f, pady=30).pack()

    labels = []
    label_frame = tk.Frame(root, bg="white")
    label_frame.pack()
    for i in range(5):
        lbl = tk.Label(label_frame, text="", bg="white", fg="#D9D9D9", font=sub_f)
        if i == 2: lbl.config(fg="#333333", font=value_f)
        lbl.pack(side="left", padx=12); labels.append(lbl)

    def update_ruler(val):
        global final_height
        final_height = float(val)
        offsets = [-STEP_HEIGHT*2, -STEP_HEIGHT, 0, STEP_HEIGHT, STEP_HEIGHT*2]
        for i, offset in enumerate(offsets):
            display_val = final_height + offset
            labels[i].config(text=f"{display_val:.1f}" if display_val >= 0 else "")

    scale = tk.Scale(root, from_=MIN_HEIGHT, to=MAX_HEIGHT, orient="horizontal", 
                     resolution=STEP_HEIGHT, showvalue=False, bg="white", bd=0, 
                     highlightthickness=0, troughcolor="#F0F0F0", length=320, command=update_ruler)
    scale.set(INIT_HEIGHT); scale.pack(pady=30)

    btn_frame = tk.Frame(root, bg="white"); btn_frame.pack()
    tk.Button(btn_frame, text="－", font=("Arial", 14), width=4, command=lambda: scale.set(scale.get()-STEP_HEIGHT)).pack(side="left", padx=20)
    tk.Button(btn_frame, text="＋", font=("Arial", 14), width=4, command=lambda: scale.set(scale.get()+STEP_HEIGHT)).pack(side="left", padx=20)

    tk.Button(root, text="저장 및 분석 시작", font=("맑은 고딕", 12, "bold"), bg="#4CAF50", fg="white", 
              width=25, height=2, command=root.destroy).pack(side="bottom", pady=40)
    root.mainloop()
    return final_height

# ==========================================
# 2. 마우스 ROI 선택 로직
# ==========================================
roi_coords, selecting = [], False
def mouse_handler(event, x, y, flags, param):
    global roi_coords, selecting, temp_f, disp_f
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_coords, selecting = [x, y], True
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        temp_f = disp_f.copy()
        cv2.rectangle(temp_f, (roi_coords[0], roi_coords[1]), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_coords.extend([x, y]); selecting = False

def get_roi_mouse(win, frame):
    global disp_f, temp_f, roi_coords
    disp_f, temp_f, roi_coords = frame.copy(), frame.copy(), []
    cv2.setMouseCallback(win, mouse_handler)
    while True:
        cv2.imshow(win, temp_f)
        if len(roi_coords) == 4:
            x1, y1, x2, y2 = roi_coords
            cv2.setMouseCallback(win, lambda *args: None)
            return (min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2))
        if cv2.waitKey(1) & 0xFF == ord('q'): return None

# ==========================================
# 3. 초기화 및 특징점 설정
# ==========================================
cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, first_frame = cap.read()
if not ret: print("영상 로드 실패"); exit()

root_info = tk.Tk(); SW, SH = root_info.winfo_screenwidth(), root_info.winfo_screenheight(); root_info.destroy()
win_setup = "DIC Setup Phase"
h_orig, w_orig = first_frame.shape[:2]
scale_factor = min(SW*0.8/w_orig, SH*0.8/h_orig, 1.0)
cv2.namedWindow(win_setup, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_setup, int(w_orig * scale_factor), int(h_orig * scale_factor))

roi_building = get_roi_mouse(win_setup, first_frame)
roi_fixed = get_roi_mouse(win_setup, first_frame)
cv2.destroyWindow(win_setup)

BUILDING_HEIGHT_M = get_ruler_picker_ui(SW, SH)
HEIGHT_CM = BUILDING_HEIGHT_M * 100

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
def preprocess(img): return clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

gray_init = preprocess(first_frame)
H_img, W_img = gray_init.shape

def find_feat(img, roi):
    x, y, w, h = roi
    c = cv2.goodFeaturesToTrack(img[y:y+h, x:x+w], 1, 0.01, 10)
    return (int(x + c[0][0][0]), int(y + c[0][0][1])) if c is not None else (x+w//2, y+h//2)

tx, ty = find_feat(gray_init, (roi_building[0], roi_building[1], roi_building[2], int(roi_building[3]*0.15)))
bx, by = find_feat(gray_init, (roi_building[0], roi_building[1] + int(roi_building[3]*0.85), roi_building[2], int(roi_building[3]*0.15)))
rx, ry = find_feat(gray_init, roi_fixed)

def get_safe_tpl(img, x, y):
    x1, y1 = max(0, x - TEMPLATE_SIZE // 2), max(0, y - TEMPLATE_SIZE // 2)
    x2, y2 = min(W_img, x1 + TEMPLATE_SIZE), min(H_img, y1 + TEMPLATE_SIZE)
    x1, y1 = max(0, x2 - TEMPLATE_SIZE), max(0, y2 - TEMPLATE_SIZE)
    return img[y1:y2, x1:x2].copy(), (x1 + TEMPLATE_SIZE//2, y1 + TEMPLATE_SIZE//2)

tpl_top, p_t0 = get_safe_tpl(gray_init, tx, ty)
tpl_bot, p_b0 = get_safe_tpl(gray_init, bx, by)
tpl_ref, p_r0 = get_safe_tpl(gray_init, rx, ry)

# [상태 관리 변수 초기화]
curr_t, prev_t = list(p_t0), list(p_t0)
curr_b, prev_b = list(p_b0), list(p_b0)
curr_r, prev_r = list(p_r0), list(p_r0)

v_pixel_dist = max(1, abs(p_t0[1] - p_b0[1]))

# 그래프 설정
plt.ion()
fig, ax1 = plt.subplots(figsize=GRAPH_FIG_SIZE)
line_cm, = ax1.plot([], [], 'tab:blue', label='Disp (cm)')
ax1.set_ylabel('Displacement (cm)', color='tab:blue', fontweight='bold')
ax2 = ax1.twinx()
line_deg, = ax2.plot([], [], 'tab:red', linestyle='--', label='Angle (deg)')
ax2.set_ylabel('Angle (deg)', color='tab:red', fontweight='bold')
ax1.set_title(f"DIC Monitoring (Building: {BUILDING_HEIGHT_M}m)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate(); ax1.grid(True, alpha=0.5)

h_cm, h_deg, t_hist = [], [], []

# ==========================================
# 5. 메인 분석 루프
# ==========================================
cv2.namedWindow("Analysis Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Analysis Monitor", int(w_orig * scale_factor), int(h_orig * scale_factor))

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = preprocess(frame)
    
    # [추적부] 로컬 탐색 + 신뢰도 검사 + 관성 적용
    def do_track(gray_img, tpl, curr, prev, update=False):
        new_pos, conf, size = track_localized(gray_img, tpl, curr, SEARCH_MARGIN)
        
        if conf >= CONFIDENCE_THRESHOLD:
            # 추적 성공: 속도 저장 및 현재 위치 갱신
            dx, dy = new_pos[0] - curr[0], new_pos[1] - curr[1]
            p_new = list(new_pos)
            
            # 템플릿 업데이트 (선택 사항)
            if update and conf > UPDATE_THRESHOLD:
                nx1, ny1 = int(p_new[0]-size[0]/2), int(p_new[1]-size[1]/2)
                if 0 <= nx1 and 0 <= ny1 and nx1+size[0] <= W_img and ny1+size[1] <= H_img:
                    new_chunk = gray_img[ny1:ny1+size[1], nx1:nx1+size[0]]
                    tpl[:] = cv2.addWeighted(tpl, 0.7, new_chunk, 0.3, 0)
            return p_new, curr, conf
        else:
            # 추적 실패: 관성(Velocity) 적용
            vx, vy = curr[0] - prev[0], curr[1] - prev[1]
            p_new = [curr[0] + vx, curr[1] + vy]
            return p_new, curr, conf

    curr_t, prev_t, conf_t = do_track(gray, tpl_top, curr_t, prev_t, update=True)
    curr_b, prev_b, conf_b = do_track(gray, tpl_bot, curr_b, prev_b, update=True)
    curr_r, prev_r, conf_r = do_track(gray, tpl_ref, curr_r, prev_r, update=False)

    # 변위 및 각도 계산 (카메라 흔들림 보정 포함)
    sx = curr_r[0] - p_r0[0]
    rm = (curr_t[0] - p_t0[0] - sx) - (curr_b[0] - p_b0[0] - sx)
    val_cm = (rm / v_pixel_dist) * HEIGHT_CM
    val_deg = np.degrees(np.arctan2(rm, v_pixel_dist))

    # 데이터 및 그래프 업데이트
    now = datetime.now()
    h_cm.append(val_cm); h_deg.append(val_deg); t_hist.append(now)
    if len(h_cm) > HISTORY_LIMIT:
        h_cm.pop(0); h_deg.pop(0); t_hist.pop(0)

    line_cm.set_data(t_hist, h_cm); line_deg.set_data(t_hist, h_deg)
    ax1.relim(); ax1.autoscale_view(); ax2.relim(); ax2.autoscale_view(); plt.pause(0.001)

    # 시각화 (고스트 오버레이)
    vis = frame.copy()
    rx, ry, rw, rh = roi_building
    roi_blended = cv2.addWeighted(frame[ry:ry+rh, rx:rx+rw], 1 - OVERLAY_ALPHA, 
                                   first_frame[ry:ry+rh, rx:rx+rw], OVERLAY_ALPHA, 0)
    vis[ry:ry+rh, rx:rx+rw] = roi_blended

    for p, c, l in [(curr_t, (255,0,0), "TOP"), (curr_b, (0,255,0), "BOT"), (curr_r, (0,0,255), "REF")]:
        pos = (int(p[0]), int(p[1]))
        cv2.drawMarker(vis, pos, c, cv2.MARKER_CROSS, MARKER_SIZE, 2)
        cv2.putText(vis, l, (pos[0]+10, pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

    cv2.putText(vis, f"Disp: {val_cm:.2f}cm | Ang: {val_deg:.3f}deg", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
    cv2.imshow("Analysis Monitor", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()