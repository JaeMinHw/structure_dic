import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import tkinter as tk
from tkinter import font as tkfont

# ==========================================
# [CONFIG] 모든 설정값 관리
# ==========================================
# VIDEO_SOURCE = 'C:/Users/admin_user/Desktop/dic_test2.mp4'
VIDEO_SOURCE = './destory_structure_video.mp4'

# 1. UI 및 높이 설정 관련
INIT_HEIGHT = 5.0      
MIN_HEIGHT = 0.1       
MAX_HEIGHT = 100.0     
STEP_HEIGHT = 0.1      

# 2. DIC 및 시각화 관련
MAX_CORNERS = 200      # 상/하단 각각 추적할 최대 지점 수
OVERLAY_ALPHA = 0.4    # 건물 영역의 첫 프레임 투명도

# 3. 그래프 데이터 관련
HISTORY_LIMIT = 200    
GRAPH_FIG_SIZE = (10, 5) 

# ==========================================
# 1. 룰러 UI (기존 코드 유지)
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
    ruler_frame = tk.Frame(root, bg="white")
    ruler_frame.pack(pady=10, fill="x")

    labels = []
    label_frame = tk.Frame(ruler_frame, bg="white")
    label_frame.pack()
    for i in range(5):
        lbl = tk.Label(label_frame, text="", bg="white", fg="#D9D9D9", font=sub_f)
        if i == 2: lbl.config(fg="#333333", font=value_f)
        lbl.pack(side="left", padx=12); labels.append(lbl)

    def update_ruler(val):
        global final_height
        final_height = float(val)
        offsets = [-0.2, -0.1, 0, 0.1, 0.2]
        for i, offset in enumerate(offsets):
            display_val = final_height + offset
            labels[i].config(text=f"{display_val:.1f}" if display_val >= 0 else "")

    scale = tk.Scale(root, from_=MIN_HEIGHT, to=MAX_HEIGHT, orient="horizontal", 
                     resolution=STEP_HEIGHT, showvalue=False, command=update_ruler)
    scale.set(INIT_HEIGHT); scale.pack(pady=20)
    tk.Button(root, text="저장 및 분석 시작", command=root.destroy, bg="#4CAF50", fg="white").pack(pady=20)
    root.mainloop()
    return final_height

# ==========================================
# 2. 마우스 ROI 선택 (기존 코드 유지)
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
            return (min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2))
        if cv2.waitKey(1) & 0xFF == ord('q'): return None

# ==========================================
# 3. 초기화 및 특징점 추출 (Robust Upgrade)
# ==========================================
cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, first_frame = cap.read()
if not ret: exit()

root_info = tk.Tk(); SW, SH = root_info.winfo_screenwidth(), root_info.winfo_screenheight(); root_info.destroy()
win_setup = "DIC Setup Phase"
h_orig, w_orig = first_frame.shape[:2]
scale_factor = 0.5 if w_orig > SW or h_orig > SH else 1.0
cv2.namedWindow(win_setup, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_setup, int(w_orig * scale_factor), int(h_orig * scale_factor))

roi_building = get_roi_mouse(win_setup, first_frame)
roi_fixed = get_roi_mouse(win_setup, first_frame)
cv2.destroyWindow(win_setup)

BUILDING_HEIGHT_M = get_ruler_picker_ui(SW, SH)
HEIGHT_CM = BUILDING_HEIGHT_M * 100
gray_init = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# [업그레이드] 구역별 다중 특징점 추출 함수
def get_multi_feats(img, roi):
    x, y, w, h = roi
    pts = cv2.goodFeaturesToTrack(img[y:y+h, x:x+w], MAX_CORNERS, 0.01, 10)
    if pts is not None:
        pts[:, 0, 0] += x
        pts[:, 0, 1] += y
        return pts
    return None

# 상단, 하단, 고정점 구역에서 수백 개의 점을 찾음
p_top_init = get_multi_feats(gray_init, (roi_building[0], roi_building[1], roi_building[2], int(roi_building[3]*0.2)))
p_bot_init = get_multi_feats(gray_init, (roi_building[0], roi_building[1] + int(roi_building[3]*0.8), roi_building[2], int(roi_building[3]*0.2)))
p_ref_init = get_multi_feats(gray_init, roi_fixed)

# 기준 거리 (상단 중앙과 하단 중앙의 거리)
v_pixel_dist = int(roi_building[3] * 0.8)

# 그래프 설정 (기존 유지)
plt.ion()
fig, ax1 = plt.subplots(figsize=GRAPH_FIG_SIZE)
line_cm, = ax1.plot([], [], color='tab:blue', label='Disp (cm)')
ax2 = ax1.twinx()
line_deg, = ax2.plot([], [], color='tab:red', linestyle='--', label='Angle (deg)')
h_cm, h_deg, t_hist = [], [], []

# ==========================================
# 4. 메인 루프 (Robust Tracking & Median Filtering)
# ==========================================
cv2.namedWindow("Analysis Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Analysis Monitor", int(w_orig * scale_factor), int(h_orig * scale_factor))

while True:
    ret, frame = cap.read()
    if not ret: break
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # [업그레이드] 수백 개의 점을 동시에 추적 (Optical Flow)
    def track_multi(p_init):
        p_curr, st, err = cv2.calcOpticalFlowPyrLK(gray_init, gray_curr, p_init, None)
        good_init = p_init[st == 1]
        good_curr = p_curr[st == 1]
        if len(good_curr) > 0:
            # 이동량 다수결(Median) 계산
            diff = good_curr - good_init
            return np.median(diff[:, 0]), np.median(diff[:, 1]), good_curr
        return 0, 0, []

    dx_t, dy_t, pts_t = track_multi(p_top_init)
    dx_b, dy_b, pts_b = track_multi(p_bot_init)
    dx_r, dy_r, pts_r = track_multi(p_ref_init)

    # 카메라 흔들림 보정 및 최종 변위 계산
    shift_x = dx_r 
    relative_move = (dx_t - shift_x) - (dx_b - shift_x)
    
    val_cm = (relative_move / v_pixel_dist) * HEIGHT_CM
    val_deg = np.degrees(np.arctan2(relative_move, v_pixel_dist))

    # 데이터 업데이트 및 그래프 (기존 유지)
    now = datetime.now()
    h_cm.append(val_cm); h_deg.append(val_deg); t_hist.append(now)
    if len(h_cm) > HISTORY_LIMIT: [h.pop(0) for h in [h_cm, h_deg, t_hist]]
    line_cm.set_data(t_hist, h_cm); line_deg.set_data(t_hist, h_deg)
    ax1.relim(); ax1.autoscale_view(); ax2.relim(); ax2.autoscale_view(); plt.pause(0.001)

    # [시각화] 고스트 효과 및 멀티 포인트 마커 (색상 구분 버전)
    ghost_display = frame.copy()
    rx, ry, rw, rh = roi_building
    roi_blended = cv2.addWeighted(frame[ry:ry+rh, rx:rx+rw], 1-OVERLAY_ALPHA, first_frame[ry:ry+rh, rx:rx+rw], OVERLAY_ALPHA, 0)
    ghost_display[ry:ry+rh, rx:rx+rw] = roi_blended

    # 1. 건물 상단 포인트 (연두색 - Green)
    for p in pts_t:
        cv2.circle(ghost_display, tuple(p.astype(int)), 2, (0, 255, 0), -1)

    # 2. 건물 하단 포인트 (노란색 - Yellow)
    for p in pts_b:
        cv2.circle(ghost_display, tuple(p.astype(int)), 2, (0, 255, 255), -1)

    # 3. 고정 기준 포인트 (하늘색 - Cyan / OpenCV에선 BGR 순서이므로 255, 255, 0)
    for p in pts_r:
        cv2.circle(ghost_display, tuple(p.astype(int)), 2, (255, 255, 0), -1)

    # 상단에 컬러 가이드 표시 (어떤 색이 무엇인지 알려주는 레전드)
    cv2.rectangle(ghost_display, (15, 15), (200, 100), (0, 0, 0), -1) # 배경 박스
    cv2.putText(ghost_display, "● Building Top", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(ghost_display, "● Building Bot", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(ghost_display, "● Fixed Ref", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 메인 수치 표시
    cv2.putText(ghost_display, f"Disp: {val_cm:.2f}cm | Ang: {val_deg:.3f}deg", (220, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Analysis Monitor", ghost_display)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()