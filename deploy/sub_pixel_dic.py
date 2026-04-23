import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import tkinter as tk
from tkinter import font as tkfont
from rembg import remove

# ==========================================
# [CONFIG] 설정값
# ==========================================
VIDEO_SOURCE = 'C:/Users/admin_user/Desktop/dic_test2.mp4'
INIT_HEIGHT, INIT_DIST = 100.0, 60.0
TEMPLATE_SIZE, MARKER_SIZE = 60, 20
OVERLAY_ALPHA, HISTORY_LIMIT = 0.6, 200
ALERT_LIMIT_CM, ALERT_LIMIT_DEG = 5.0, 1.0

# ==========================================
# 1. 모니터 해상도 감지 (창을 크게 띄우기 위함)
# ==========================================
def get_monitor_size():
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return sw, sh

# ==========================================
# 2. 초기 세팅 및 ROI 선택
# ==========================================
cap = cv2.VideoCapture(VIDEO_SOURCE); ret, first_frame = cap.read()
if not ret: exit()

h_orig, w_orig = first_frame.shape[:2]
SW, SH = get_monitor_size()

# 창 크기 결정 (모니터의 95% 크기로 시원하게 설정)
scale_for_ui = min(SW * 0.95 / w_orig, SH * 0.95 / h_orig, 1.0)

# [UI 실행]
def get_config_ui(sw, sh):
    root = tk.Tk(); root.title("측정 환경 설정")
    root.geometry(f"420x650+{int(sw/2-210)}+{int(sh/2-325)}"); root.configure(bg="white")
    res = [INIT_DIST, INIT_HEIGHT]; step = [1]
    lbl_t = tk.Label(root, text="Step 1: 건물과 카메라 거리", bg="white", font=("맑은 고딕", 14, "bold"), pady=30); lbl_t.pack()
    lbl_v = tk.Label(root, text=f"{res[0]:.1f}", font=("Arial", 35, "bold"), bg="white"); lbl_v.pack()
    sc = tk.Scale(root, from_=0.1, to=500.0, orient="horizontal", resolution=0.1, showvalue=False, length=300, command=lambda v: lbl_v.config(text=f"{float(v):.1f}"))
    sc.set(res[0]); sc.pack(pady=40)
    def nxt():
        if step[0] == 1: res[0]=sc.get(); step[0]=2; lbl_t.config(text="Step 2: 건물의 실제 높이"); sc.set(res[1])
        else: res[1]=sc.get(); root.destroy()
    tk.Button(root, text="다음", bg="#4CAF50", fg="white", width=20, height=2, command=nxt).pack(side="bottom", pady=50)
    root.mainloop(); return res

DIST_M, HEIGHT_M = get_config_ui(SW, SH)

# ROI 선택 (드래그 방식)
roi_coords, selecting = [], False
def mouse_handler(e, x, y, f, p):
    global roi_coords, selecting, temp_f, disp_f
    if e == cv2.EVENT_LBUTTONDOWN: roi_coords, selecting = [x, y], True
    elif e == cv2.EVENT_MOUSEMOVE and selecting:
        temp_f = disp_f.copy()
        cv2.rectangle(temp_f, (roi_coords[0], roi_coords[1]), (x, y), (0, 255, 0), 2)
    elif e == cv2.EVENT_LBUTTONUP: roi_coords.extend([x, y]); selecting = False

def get_roi(win, frame):
    global disp_f, temp_f, roi_coords
    # 선택할 때는 편의상 화면 크기에 맞춤
    disp_f = cv2.resize(frame, (int(w_orig * scale_for_ui), int(h_orig * scale_for_ui)), interpolation=cv2.INTER_AREA)
    temp_f = disp_f.copy(); roi_coords = []
    cv2.setMouseCallback(win, mouse_handler)
    while True:
        cv2.imshow(win, temp_f)
        if len(roi_coords) == 4:
            # 원본 좌표로 정밀하게 환산
            x1, y1, x2, y2 = [int(c / scale_for_ui) for c in roi_coords]
            return (min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2))
        if cv2.waitKey(1) == ord('q'): exit()

win_s = "Setup (Drag to Select)"
cv2.namedWindow(win_s, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_s, int(w_orig * scale_for_ui), int(h_orig * scale_for_ui))
b_x, b_y, b_w, b_h = get_roi(win_s, first_frame)
f_roi = get_roi(win_s, first_frame)
cv2.destroyWindow(win_s)

# 4. rembg 누끼 (원본 화질)
print("AI 누끼 작업 중..."); bc = first_frame[b_y:b_y+b_h, b_x:b_x+b_w]; nk = remove(bc)
gp = nk[:, :, :3]; ga = (nk[:, :, 3] / 255.0)[:, :, np.newaxis] * OVERLAY_ALPHA

# 5. 서브픽셀 분석 로직
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
def preprocess(img): return clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
gray_init = preprocess(first_frame)
def find_feat(img, r):
    x, y, w, h = r; c = cv2.goodFeaturesToTrack(img[y:y+h, x:x+w], 1, 0.01, 10)
    return (int(x + c[0][0][0]), int(y + c[0][0][1])) if c is not None else (x+w//2, y+h//2)
tx, ty = find_feat(gray_init, (b_x, b_y, b_w, int(b_h*0.15)))
bx, by = find_feat(gray_init, (b_x, b_y+int(b_h*0.85), b_w, int(b_h*0.15)))
rx, ry = find_feat(gray_init, f_roi)
def get_tpl(img, x, y):
    x1, y1 = max(1, x-TEMPLATE_SIZE//2), max(1, y-TEMPLATE_SIZE//2)
    return img[y1:y1+TEMPLATE_SIZE, x1:x1+TEMPLATE_SIZE], (x1+TEMPLATE_SIZE//2, y1+TEMPLATE_SIZE//2)
tpl_t, p_t0 = get_tpl(gray_init, tx, ty); tpl_b, p_b0 = get_tpl(gray_init, bx, by)
tpl_r, p_r0 = get_tpl(gray_init, rx, ry); v_dist = max(1, abs(p_t0[1]-p_b0[1]))
TILT = (np.sqrt(DIST_M**2+HEIGHT_M**2)/np.sqrt(DIST_M**2+1.5**2))**2

def track_subpixel(img, tpl):
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res); x0, y0 = max_loc
    if 0 < x0 < res.shape[1]-1 and 0 < y0 < res.shape[0]-1:
        dx = (res[y0, x0-1] - res[y0, x0+1]) / (2 * (res[y0, x0-1] - 2*res[y0, x0] + res[y0, x0+1]) + 1e-5)
        dy = (res[y0-1, x0] - res[y0+1, x0]) / (2 * (res[y0-1, x0] - 2*res[y0, x0] + res[y0+1, x0]) + 1e-5)
        return x0 + tpl.shape[1]//2 + dx, y0 + tpl.shape[0]//2 + dy
    return x0 + tpl.shape[1]//2, y0 + tpl.shape[0]//2

# 그래프 설정 (디테일 유지)
plt.ion(); fig, ax1 = plt.subplots(figsize=(10, 5))
color_cm, color_deg = 'tab:blue', 'tab:red'
ax1.set_xlabel('Time'); ax1.set_ylabel('Disp (cm)', color=color_cm, fontweight='bold')
line_cm, = ax1.plot([], [], color=color_cm, linewidth=1.5); ax1.tick_params(axis='y', labelcolor=color_cm)
ax2 = ax1.twinx(); ax2.set_ylabel('Angle (deg)', color=color_deg, fontweight='bold')
line_deg, = ax2.plot([], [], color=color_deg, linestyle='--'); ax2.tick_params(axis='y', labelcolor=color_deg)
ax1.grid(True, linestyle='--', alpha=0.5); h_cm, h_deg, t_hist = [], [], []

# ==========================================
# 6. 메인 분석 루프 (고해상도 창 설정)
# ==========================================
cv2.namedWindow("Monitor", cv2.WINDOW_NORMAL) # 창 크기 조절 가능 모드
# 시작할 때 모니터 해상도에 맞춰 창 크기 조정 (이미지는 원본 해상도 유지)
cv2.resizeWindow("Monitor", int(w_orig * scale_for_ui), int(h_orig * scale_for_ui))

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # [핵심] 100% 원본 해상도 이미지에서 작업
    gd = frame.copy(); rc = frame[b_y:b_y+b_h, b_x:b_x+b_w]
    if rc.shape[:2] == gp.shape[:2]:
        # 라이브 영상 위에 누끼 잔상 합성
        gd[b_y:b_y+b_h, b_x:b_x+b_w] = (rc*(1-ga) + gp*ga).astype('uint8')
    
    gray = preprocess(frame)
    ct, cb, cr = track_subpixel(gray, tpl_t), track_subpixel(gray, tpl_b), track_subpixel(gray, tpl_r)
    
    sx = cr[0] - p_r0[0]
    rm = ((ct[0] - p_t0[0] - sx) * TILT) - (cb[0] - p_b0[0] - sx)
    val_cm = (rm / v_dist) * (HEIGHT_M * 100)
    val_deg = np.degrees(np.arctan2(val_cm / 100.0, HEIGHT_M))

    now = datetime.now(); h_cm.append(val_cm); h_deg.append(val_deg); t_hist.append(now)
    if len(h_cm) > HISTORY_LIMIT: h_cm.pop(0); h_deg.pop(0); t_hist.pop(0)
    line_cm.set_data(t_hist, h_cm); line_deg.set_data(t_hist, h_deg)
    ax1.relim(); ax1.autoscale_view(); ax2.relim(); ax2.autoscale_view(); plt.pause(0.001)

    # 마커 표시 (원본 좌표 그대로 사용해서 정밀함)
    for p, c in [(ct, (255,0,0)), (cb, (0,255,0)), (cr, (0,0,255))]:
        cv2.drawMarker(gd, (int(p[0]), int(p[1])), c, cv2.MARKER_CROSS, MARKER_SIZE, 2)
    
    # 경고 표시 (글자 크기를 이미지 크기에 맞게 조정)
    if abs(val_cm) >= ALERT_LIMIT_CM:
        font_scale = max(1.5, w_orig / 1000) # 해상도에 맞춰 글자 크기 조절
        cv2.putText(gd, "DANGER!", (int(w_orig*0.4), int(h_orig*0.5)), 2, font_scale, (0,0,255), 3)
        cv2.rectangle(gd, (0,0), (w_orig, h_orig), (0,0,255), 20)

    # 창에 출력 (OpenCV가 알아서 창 크기에 맞춰 보여줌)
    cv2.imshow("Monitor", gd)
    
    if cv2.waitKey(1) == ord('q'): break

cap.release(); cv2.destroyAllWindows()