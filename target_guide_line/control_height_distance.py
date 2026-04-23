# 이 코드는 ai로 누끼를 따서 올리는건데 건물로 했을 때 잘 되는지는 모르겠음. 또한 다시 원점으로 돌렸을 때 0으로 가야하는데 0으로 가지 않음. 단일 점. 빨간색 알림 창

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import tkinter as tk
from tkinter import font as tkfont
from rembg import remove

# ==========================================
# [CONFIG] 모든 설정값 관리
# ==========================================
VIDEO_SOURCE = 'C:/Users/admin_user/Desktop/dic_test2.mp4'

# 1. 환경 및 DIC 설정
INIT_HEIGHT = 100.0    
INIT_DIST = 60.0       
TEMPLATE_SIZE = 60     
MARKER_SIZE = 20       
OVERLAY_ALPHA = 0.6    
HISTORY_LIMIT = 200    
GRAPH_FIG_SIZE = (10, 5) 

# 2. [ALERT] 경고 임계값 (원하는 수치로 조정하세요)
ALERT_LIMIT_CM = 0.5   # 5cm 이상 시 경고
ALERT_LIMIT_DEG = 1.0  # 1도 이상 시 경고

# ==========================================
# 1. 환경 설정 UI (직관적인 문구 적용)
# ==========================================
conf_dist, conf_height = INIT_DIST, INIT_HEIGHT

def get_config_ui(screen_w, screen_h):
    global conf_dist, conf_height
    root = tk.Tk()
    root.title("측정 환경 설정")
    win_w, win_h = 420, 650
    root.geometry(f"{win_w}x{win_h}+{int(screen_w/2-win_w/2)}+{int(screen_h/2-win_h/2)}")
    root.configure(bg="white")
    title_f = tkfont.Font(family="맑은 고딕", size=14, weight="bold")
    val_f = tkfont.Font(family="Arial", size=35, weight="bold")
    current_step = [1]
    title_lbl = tk.Label(root, text="Step 1: 건물과 카메라 사이의 거리", bg="white", font=title_f, pady=30)
    title_lbl.pack()
    val_lbl = tk.Label(root, text=f"{INIT_DIST:.1f}", font=val_f, bg="white", fg="#333333")
    val_lbl.pack()
    scale = tk.Scale(root, from_=0.1, to=500.0, orient="horizontal", resolution=0.1, showvalue=False, bg="white", length=300, command=lambda v: val_lbl.config(text=f"{float(v):.1f}"))
    scale.set(INIT_DIST)
    scale.pack(pady=40)
    def on_next():
        if current_step[0] == 1:
            global conf_dist
            conf_dist = scale.get()
            current_step[0] = 2
            title_lbl.config(text="Step 2: 건물의 실제 높이")
            scale.set(INIT_HEIGHT)
            next_btn.config(text="저장 및 분석 시작")
        else:
            global conf_height
            conf_height = scale.get()
            root.destroy()
    next_btn = tk.Button(root, text="다음 단계", font=("맑은 고딕", 12, "bold"), bg="#4CAF50", fg="white", width=25, height=2, command=on_next)
    next_btn.pack(side="bottom", pady=50)
    root.mainloop()
    return conf_dist, conf_height

# ==========================================
# 2. 초기 세팅 및 ROI 선택
# ==========================================
cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, first_frame = cap.read()
if not ret: exit()
h_orig, w_orig = first_frame.shape[:2]
root_info = tk.Tk(); SW, SH = root_info.winfo_screenwidth(), root_info.winfo_screenheight(); root_info.destroy()
DIST_M, HEIGHT_M = get_config_ui(SW, SH)

roi_coords, selecting = [], False
def mouse_handler(event, x, y, flags, param):
    global roi_coords, selecting, temp_f, disp_f
    if event == cv2.EVENT_LBUTTONDOWN: roi_coords, selecting = [x, y], True
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        temp_f = disp_f.copy()
        cv2.rectangle(temp_f, (roi_coords[0], roi_coords[1]), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP: roi_coords.extend([x, y]); selecting = False

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
        if cv2.waitKey(1) & 0xFF == ord('q'): exit()

win_setup = "DIC Setup Phase"
scale_factor = 0.5 if w_orig > SW or h_orig > SH else 1.0
cv2.namedWindow(win_setup, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_setup, int(w_orig * scale_factor), int(h_orig * scale_factor))
b_x, b_y, b_w, b_h = get_roi_mouse(win_setup, first_frame)
f_roi = get_roi_mouse(win_setup, first_frame)
cv2.destroyWindow(win_setup)

# 4. rembg 누끼 작업
print("AI 누끼 작업 중..."); building_crop = first_frame[b_y:b_y+b_h, b_x:b_x+b_w]
nooki_rgba = remove(building_crop); ghost_patch = nooki_rgba[:, :, :3]
ghost_alpha = (nooki_rgba[:, :, 3] / 255.0)[:, :, np.newaxis] * OVERLAY_ALPHA

# 5. DIC 분석 및 보정 수식 준비
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
def preprocess(img): return clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
gray_init = preprocess(first_frame)
def find_feat(img, roi):
    x, y, w, h = roi
    c = cv2.goodFeaturesToTrack(img[y:y+h, x:x+w], 1, 0.01, 10)
    return (int(x + c[0][0][0]), int(y + c[0][0][1])) if c is not None else (x+w//2, y+h//2)
tx, ty = find_feat(gray_init, (b_x, b_y, b_w, int(b_h*0.15)))
bx, by = find_feat(gray_init, (b_x, b_y + int(b_h*0.85), b_w, int(b_h*0.15)))
ref_x, ref_y = find_feat(gray_init, f_roi)
def get_safe_tpl(img, x, y):
    x1, y1 = max(0, x - TEMPLATE_SIZE // 2), max(0, y - TEMPLATE_SIZE // 2)
    x2, y2 = min(w_orig, x1 + TEMPLATE_SIZE), min(h_orig, y1 + TEMPLATE_SIZE)
    x1, y1 = max(0, x2 - TEMPLATE_SIZE), max(0, y2 - TEMPLATE_SIZE)
    return img[y1:y2, x1:x2], (x1 + TEMPLATE_SIZE//2, y1 + TEMPLATE_SIZE//2)
tpl_top, p_t0 = get_safe_tpl(gray_init, tx, ty); tpl_bot, p_b0 = get_safe_tpl(gray_init, bx, by)
tpl_ref, p_r0 = get_safe_tpl(gray_init, ref_x, ref_y); v_pixel_dist = max(1, abs(p_t0[1] - p_b0[1]))
L_top, L_bot = np.sqrt(DIST_M**2 + HEIGHT_M**2), np.sqrt(DIST_M**2 + 1.5**2)
TILT_CORRECTION = (L_top / L_bot)**2 

# ==========================================
# 4. 실시간 듀얼 그래프 설정 (원상복구 버전)
# ==========================================
plt.ion()
fig, ax1 = plt.subplots(figsize=GRAPH_FIG_SIZE)
color_cm = 'tab:blue'
ax1.set_xlabel('Current Time')
ax1.set_ylabel('Displacement (cm)', color=color_cm, fontweight='bold')
line_cm, = ax1.plot([], [], color=color_cm, linewidth=1.5, label='Disp (cm)')
ax1.tick_params(axis='y', labelcolor=color_cm)

ax2 = ax1.twinx()
color_deg = 'tab:red'
ax2.set_ylabel('Angle (deg)', color=color_deg, fontweight='bold')
line_deg, = ax2.plot([], [], color=color_deg, linewidth=1.5, linestyle='--', label='Angle (deg)')
ax2.tick_params(axis='y', labelcolor=color_deg)

ax1.set_title(f"DIC AI Monitoring (D:{DIST_M}m, H:{HEIGHT_M}m)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()
ax1.grid(True, linestyle='--', alpha=0.5)

h_cm, h_deg, t_hist = [], [], []

# ==========================================
# 6. 메인 분석 루프 (경고 시스템 포함)
# ==========================================
cv2.namedWindow("Analysis Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Analysis Monitor", int(w_orig * scale_factor), int(h_orig * scale_factor))

while True:
    ret, frame = cap.read()
    if not ret: break
    
    ghost_display = frame.copy()
    roi_current = frame[b_y:b_y+b_h, b_x:b_x+b_w]
    if roi_current.shape[:2] == ghost_patch.shape[:2]:
        roi_blended = (roi_current * (1 - ghost_alpha) + ghost_patch * ghost_alpha).astype('uint8')
        ghost_display[b_y:b_y+b_h, b_x:b_x+b_w] = roi_blended
    
    gray = preprocess(frame)
    def track(img, tpl):
        res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(res)
        return loc[0] + tpl.shape[1]//2, loc[1] + tpl.shape[0]//2
    ct, cb, cr = track(gray, tpl_top), track(gray, tpl_bot), track(gray, tpl_ref)
    
    sx = cr[0] - p_r0[0]
    rm = ((ct[0] - p_t0[0] - sx) * TILT_CORRECTION) - (cb[0] - p_b0[0] - sx)
    val_cm = (rm / v_pixel_dist) * (HEIGHT_M * 100)
    val_deg = np.degrees(np.arctan2(val_cm / 100.0, HEIGHT_M))

    # [경고 체크]
    is_alert = abs(val_cm) >= ALERT_LIMIT_CM or abs(val_deg) >= ALERT_LIMIT_DEG

    # 데이터 및 그래프 업데이트
    now = datetime.now(); h_cm.append(val_cm); h_deg.append(val_deg); t_hist.append(now)
    if len(h_cm) > HISTORY_LIMIT: h_cm.pop(0); h_deg.pop(0); t_hist.pop(0)
    line_cm.set_data(t_hist, h_cm); line_deg.set_data(t_hist, h_deg)
    ax1.relim(); ax1.autoscale_view(); ax2.relim(); ax2.autoscale_view(); plt.pause(0.001)

    # 시각화 (마커 및 경고 오버레이)
    for p, c in [(ct, (255,0,0)), (cb, (0,255,0)), (cr, (0,0,255))]:
        cv2.drawMarker(ghost_display, p, c, cv2.MARKER_CROSS, MARKER_SIZE, 2)
    
    if is_alert:
        # 화면 테두리 및 경고 텍스트 (분석은 멈추지 않음)
        cv2.rectangle(ghost_display, (0, 0), (w_orig, h_orig), (0, 0, 255), 25)
        cv2.putText(ghost_display, "!!! DANGER: LIMIT EXCEEDED !!!", (int(w_orig*0.25), int(h_orig*0.5)), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 255), 4)
        cv2.putText(ghost_display, f"Disp: {val_cm:.2f}cm", (int(w_orig*0.35), int(h_orig*0.58)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    else:
        cv2.putText(ghost_display, f"Status: Normal | Disp: {val_cm:.2f}cm", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Analysis Monitor", ghost_display)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()