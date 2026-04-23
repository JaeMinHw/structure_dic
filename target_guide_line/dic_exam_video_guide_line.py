# 파이썬 내장 누끼로 적용한 코드. 원점으로 돌아왔을 때 0으로 오기는하지만 누끼 부분이 부족. 단일 점

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
VIDEO_SOURCE = 'C:/Users/admin_user/Desktop/dic_test2.mp4'

# 1. UI 및 높이 설정 관련
INIT_HEIGHT = 5.0      
MIN_HEIGHT = 0.1       
MAX_HEIGHT = 100.0     
STEP_HEIGHT = 0.1      

# 2. DIC 및 시각화 관련
TEMPLATE_SIZE = 60     
MARKER_SIZE = 20       
OVERLAY_ALPHA = 0.4    # [핵심] 누끼 딴 건물 잔상의 투명도 (0.4 = 40% 잔상)

# 3. 그래프 및 데이터 관련
HISTORY_LIMIT = 200    
GRAPH_FIG_SIZE = (10, 5) 

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

    ruler_frame = tk.Frame(root, bg="white")
    ruler_frame.pack(pady=10, fill="x")

    labels = []
    label_frame = tk.Frame(ruler_frame, bg="white")
    label_frame.pack()

    for i in range(5):
        lbl = tk.Label(label_frame, text="", bg="white", fg="#D9D9D9", font=sub_f)
        if i == 2: lbl.config(fg="#333333", font=value_f)
        lbl.pack(side="left", padx=12)
        labels.append(lbl)

    tk.Label(root, text="meters", font=unit_f, bg="white", fg="gray").pack()

    def update_ruler(val):
        global final_height
        final_height = float(val)
        offsets = [-STEP_HEIGHT*2, -STEP_HEIGHT, 0, STEP_HEIGHT, STEP_HEIGHT*2]
        for i, offset in enumerate(offsets):
            display_val = final_height + offset
            if display_val < 0: labels[i].config(text="")
            else: labels[i].config(text=f"{display_val:.1f}")

    style_frame = tk.Frame(root, bg="white", pady=30)
    style_frame.pack(fill="x", padx=40)

    scale = tk.Scale(style_frame, from_=MIN_HEIGHT, to=MAX_HEIGHT, orient="horizontal", 
                     resolution=STEP_HEIGHT, showvalue=False, bg="white", bd=0, 
                     highlightthickness=0, troughcolor="#F0F0F0", activebackground="#4CAF50",
                     length=320, command=update_ruler)
    scale.set(INIT_HEIGHT)
    scale.pack()

    btn_frame = tk.Frame(root, bg="white")
    btn_frame.pack(pady=10)

    def adjust(amount):
        new_val = round(scale.get() + amount, 2)
        if MIN_HEIGHT <= new_val <= MAX_HEIGHT: scale.set(new_val)

    tk.Button(btn_frame, text="－", font=("Arial", 14), width=4, command=lambda: adjust(-STEP_HEIGHT), bg="#F8F8F8", bd=1).pack(side="left", padx=20)
    tk.Button(btn_frame, text="＋", font=("Arial", 14), width=4, command=lambda: adjust(STEP_HEIGHT), bg="#F8F8F8", bd=1).pack(side="left", padx=20)

    tk.Button(root, text="저장 및 분석 시작", font=("맑은 고딕", 12, "bold"), bg="#4CAF50", fg="white", 
              width=25, height=2, bd=0, command=root.destroy, cursor="hand2").pack(side="bottom", pady=40)

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
# 3. 전처리 및 초기화 단계 (자동 누끼 포함)
# ==========================================
cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, first_frame = cap.read()
if not ret: print("Error: 영상을 불러올 수 없습니다."); exit()

root_info = tk.Tk(); SW, SH = root_info.winfo_screenwidth(), root_info.winfo_screenheight(); root_info.destroy()

win_setup = "DIC Setup Phase"
h_orig, w_orig = first_frame.shape[:2]
scale_factor = 0.5 if w_orig > SW or h_orig > SH else 1.0
cv2.namedWindow(win_setup, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_setup, int(w_orig * scale_factor), int(h_orig * scale_factor))

# 영역 선택
print("1단계: 건물(추적 대상) 영역을 드래그하세요.")
roi_building = get_roi_mouse(win_setup, first_frame)
print("2단계: 카메라 흔들림 보정을 위한 고정점 영역을 드래그하세요.")
roi_fixed = get_roi_mouse(win_setup, first_frame)
cv2.destroyWindow(win_setup)

# 높이 입력 (마우스 룰러 UI)
BUILDING_HEIGHT_M = get_ruler_picker_ui(SW, SH)
HEIGHT_CM = BUILDING_HEIGHT_M * 100

print("설정 완료. 분석 및 건물 누끼 작업을 시작합니다...")

# ==========================================
# [신규] GrabCut 알고리즘을 이용한 건물 자동 배경 제거
# ==========================================
# 드래그한 직사각형 영역을 기반으로 건물 객체만 분리하는 마스크를 생성합니다.
mask = np.zeros(first_frame.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
# 사용자 드래그 좌표를 GrabCut 형식에 맞게 변환
# GrabCut은 (x, y, w, h) 형식을 사용합니다.
cv2.grabCut(first_frame, mask, roi_building, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 확실한 전경(3)과 아마 전경(1)을 전경(1), 나머지를 배경(0)으로 만드는 마스크
# [image_0.png, image_1.png]에서 볼 수 있듯이 마스크를 통해 객체만 추출합니다.
building_mask = np.where((mask==cv2.GC_PR_FGD)|(mask==cv2.GC_FGD), 1, 0).astype('uint8')

# 첫 프레임에서 건물만 추출한 '고스트 패치' 생성
# [image_2.png, image_3.png]와 같은 투명한 건물 이미지를 만드는 과정입니다.
rx, ry, rw, rh = roi_building
ghost_building_patch = first_frame[ry:ry+rh, rx:rx+rw] * building_mask[ry:ry+rh, rx:rx+rw, np.newaxis]
ghost_alpha_mask = building_mask[ry:ry+rh, rx:rx+rw] * OVERLAY_ALPHA # 투명도 적용된 마스크

# ==========================================
# 4. DIC 연산 준비 및 그래프 설정 (기존과 동일)
# ==========================================
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
    return img[y1:y2, x1:x2], (x1 + TEMPLATE_SIZE//2, y1 + TEMPLATE_SIZE//2)
tpl_top, p_t0 = get_safe_tpl(gray_init, tx, ty)
tpl_bot, p_b0 = get_safe_tpl(gray_init, bx, by)
tpl_ref, p_r0 = get_safe_tpl(gray_init, rx, ry)
v_pixel_dist = max(1, abs(p_t0[1] - p_b0[1]))

plt.ion()
fig, ax1 = plt.subplots(figsize=GRAPH_FIG_SIZE)
color_cm = 'tab:blue'; ax1.set_xlabel('Current Time'); ax1.set_ylabel('Displacement (cm)', color=color_cm, fontweight='bold')
line_cm, = ax1.plot([], [], color=color_cm, linewidth=1.5, label='Disp (cm)'); ax1.tick_params(axis='y', labelcolor=color_cm)
ax2 = ax1.twinx(); color_deg = 'tab:red'; ax2.set_ylabel('Angle (deg)', color=color_deg, fontweight='bold')
line_deg, = ax2.plot([], [], color=color_deg, linewidth=1.5, linestyle='--', label='Angle (deg)'); ax2.tick_params(axis='y', labelcolor=color_deg)
ax1.set_title(f"DIC Object Monitoring (H: {BUILDING_HEIGHT_M}m)"); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S')); fig.autofmt_xdate(); ax1.grid(True, linestyle='--', alpha=0.5)
h_cm, h_deg, t_hist = [], [], []

# ==========================================
# 5. 메인 분석 루프 (건물 '누끼' 전용 오버레이)
# ==========================================
cv2.namedWindow("Analysis Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Analysis Monitor", int(w_orig * scale_factor), int(h_orig * scale_factor))

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # [핵심] 건물 전용 마스크 합성 (Masked Blending)
    # 주변 배경은 100% 선명하게 유지하고, 건물 모양대로만 과거 잔상을 합성합니다.
    # [image_4.png]의 방식과 유사하게 마스크를 기준으로 합성합니다.
    ghost_display = frame.copy()
    rx, ry, rw, rh = roi_building
    roi_current = frame[ry:ry+rh, rx:rx+rw]
    
    # 합성 공식: (현재 영역 * (1 - 마스크)) + (과거 건물 누끼 이미지 * 마스크)
    # 마스크가 있는 건물 영역에만 과거 잔상이 '덮어씌워지는' 효과를 냅니다.
    roi_blended = (roi_current * (1 - ghost_alpha_mask[:, :, np.newaxis]) + \
                  ghost_building_patch * ghost_alpha_mask[:, :, np.newaxis]).astype('uint8')
    
    # 합성된 영역을 현재 화면에 다시 적용
    ghost_display[ry:ry+rh, rx:rx+rw] = roi_blended
    
    # DIC 연산
    gray = preprocess(frame)
    def track(img, tpl):
        res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(res)
        return loc[0] + tpl.shape[1]//2, loc[1] + tpl.shape[0]//2
    ct, cb, cr = track(gray, tpl_top), track(gray, tpl_bot), track(gray, tpl_ref)
    
    sx = cr[0] - p_r0[0]
    rm = (ct[0] - p_t0[0] - sx) - (cb[0] - p_b0[0] - sx)
    val_cm = (rm / v_pixel_dist) * HEIGHT_CM
    val_deg = np.degrees(np.arctan2(rm, v_pixel_dist))

    now = datetime.now(); h_cm.append(val_cm); h_deg.append(val_deg); t_hist.append(now)
    if len(h_cm) > HISTORY_LIMIT: h_cm.pop(0); h_deg.pop(0); t_hist.pop(0)
    line_cm.set_data(t_hist, h_cm); line_deg.set_data(t_hist, h_deg); ax1.relim(); ax1.autoscale_view(); ax2.relim(); ax2.autoscale_view(); plt.pause(0.001)

    # 시각화 (누끼 오버레이 화면 ghost_display 위에 그리기)
    for p, c in [(ct, (255,0,0)), (cb, (0,255,0)), (cr, (0,0,255))]:
        cv2.drawMarker(ghost_display, p, c, cv2.MARKER_CROSS, MARKER_SIZE, 2)
    cv2.putText(ghost_display, f"Disp: {val_cm:.2f}cm | Ang: {val_deg:.3f}deg", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
    cv2.imshow("Analysis Monitor", ghost_display)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()