import matplotlib
matplotlib.use('TkAgg') # Tkinter 임베딩을 위한 백엔드 고정
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import cv2
import numpy as np
import tkinter as tk
from tkinter import font as tkfont
from datetime import datetime
import os

# ==========================================
# [CONFIG] 시스템 설정
# ==========================================
VIDEO_SOURCE = './destory_structure_video.mp4'
TEMPLATE_SIZE = 120    
SEARCH_MARGIN = 100
CONFIDENCE_THRESHOLD = 0.7 
UPDATE_THRESHOLD = 0.90     
MARKER_SIZE = 20       
HEATMAP_ALPHA = 0.6    
GHOST_ALPHA = 0.4      
HISTORY_LIMIT = 200    

# --- 분석 로직 함수들 ---
def track_ghost_logic(img, tpl, prev_pos, margin):
    px, py = int(prev_pos[0]), int(prev_pos[1]); th, tw = tpl.shape[:2]
    x1, y1 = max(0, px - tw//2 - margin), max(0, py - th//2 - margin)
    x2, y2 = min(img.shape[1], px + tw//2 + margin), min(img.shape[0], py + th//2 + margin)
    search_area = img[y1:y2, x1:x2]
    if search_area.shape[0] < th or search_area.shape[1] < tw: return prev_pos, 0.0
    res = cv2.matchTemplate(search_area, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return (x1 + max_loc[0] + tw/2, y1 + max_loc[1] + th/2), max_val

def track_heat_logic(init_pos, warp_matrix):
    px, py = init_pos
    new_x = warp_matrix[0, 0] * px + warp_matrix[0, 1] * py + warp_matrix[0, 2]
    new_y = warp_matrix[1, 0] * px + warp_matrix[1, 1] * py + warp_matrix[1, 2]
    return (new_x, new_y)

# ==========================================
# 2. UI 및 ROI 선택 (기존 스타일 완벽 복구)
# ==========================================
cv2.destroyAllWindows()
root_main = tk.Tk()
SW, SH = root_main.winfo_screenwidth(), root_main.winfo_screenheight()
root_main.withdraw() # 메인 윈도우는 관리용으로만 사용

# 4등분 크기 계산
W_4, H_4 = int(SW / 2), int(SH / 2) - 80

cap = cv2.VideoCapture(VIDEO_SOURCE); ret, first_frame = cap.read()
if not ret: exit()

# [1단계] ROI 드래그 선택 (2번)
win_setup = "DIC Setup"
cv2.namedWindow(win_setup, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_setup, int(SW*0.7), int(SH*0.7))

roi_coords, selecting = [], False
def mouse_handler(event, x, y, flags, param):
    global roi_coords, selecting, temp_f, disp_f
    if event == cv2.EVENT_LBUTTONDOWN: roi_coords, selecting = [x, y], True
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        temp_f = disp_f.copy(); cv2.rectangle(temp_f, (roi_coords[0], roi_coords[1]), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP: roi_coords.extend([x, y]); selecting = False

def get_roi_mouse(win, frame):
    global disp_f, temp_f, roi_coords
    disp_f, temp_f, roi_coords = frame.copy(), frame.copy(), []
    cv2.setMouseCallback(win, mouse_handler)
    while True:
        cv2.imshow(win, temp_f)
        if len(roi_coords) == 4:
            return (min(roi_coords[0], roi_coords[2]), min(roi_coords[1], roi_coords[3]), abs(roi_coords[0]-roi_coords[2]), abs(roi_coords[1]-roi_coords[3]))
        if cv2.waitKey(1) & 0xFF == ord('q'): exit()

print("1단계: 건물(히트맵) 영역 드래그")
roi_building = get_roi_mouse(win_setup, first_frame)
print("2단계: 고정 참조점 영역 드래그")
roi_fixed = get_roi_mouse(win_setup, first_frame)
cv2.destroyWindow(win_setup)

# [2단계] 원래 쓰시던 높이 조절 슬라이더 UI 복구
final_height = 5.0
def get_ruler_picker_ui():
    global final_height
    h_win = tk.Toplevel(root_main); h_win.title("구조물 높이 설정")
    h_win.geometry(f"400x500+{(SW-400)//2}+{(SH-500)//2}")
    tk.Label(h_win, text="구조물의 실제 높이(m) 입력", font=("맑은 고딕", 14, "bold")).pack(pady=30)
    v_lbl = tk.Label(h_win, text=f"{final_height:.1f}", font=("Arial", 45, "bold")); v_lbl.pack()
    def update_val(v): global final_height; final_height = float(v); v_lbl.config(text=f"{final_height:.1f}")
    scale = tk.Scale(h_win, from_=0.1, to=100.0, orient="horizontal", resolution=0.1, length=300, showvalue=False, command=update_val)
    scale.set(final_height); scale.pack(pady=20)
    tk.Button(h_win, text="저장 후 분석 시작", bg="#4CAF50", fg="white", font=("맑은 고딕", 12, "bold"), width=20, height=2, command=h_win.destroy).pack(pady=40)
    h_win.grab_set(); root_main.wait_window(h_win)

get_ruler_picker_ui(); HEIGHT_CM = final_height * 100

# [3단계] 분석 특징점 초기화
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
def preprocess(img): return clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
gray_init = preprocess(first_frame)

def find_feat(img, roi):
    x, y, w, h = roi; c = cv2.goodFeaturesToTrack(img[y:y+h, x:x+w], 1, 0.01, 10)
    return (int(x + c[0][0][0]), int(y + c[0][0][1])) if c is not None else (x+w//2, y+h//2)

tx, ty = find_feat(gray_init, (roi_building[0], roi_building[1], roi_building[2], int(roi_building[3]*0.2)))
bx, by = find_feat(gray_init, (roi_building[0], roi_building[1] + int(roi_building[3]*0.8), roi_building[2], int(roi_building[3]*0.2)))
rx, ry = find_feat(gray_init, roi_fixed)

def get_tpl(img, x, y):
    x1, y1 = max(0, x-TEMPLATE_SIZE//2), max(0, y-TEMPLATE_SIZE//2)
    return img[y1:y1+TEMPLATE_SIZE, x1:x1+TEMPLATE_SIZE].copy(), (x1+TEMPLATE_SIZE//2, y1+TEMPLATE_SIZE//2)

tpl_top, p_t0 = get_tpl(gray_init, tx, ty); tpl_bot, p_b0 = get_tpl(gray_init, bx, by); tpl_ref, p_r0 = get_tpl(gray_init, rx, ry)
curr_t_ghost, prev_t_ghost, curr_b_ghost, prev_b_ghost, curr_r_ghost, prev_r_ghost = list(p_t0), list(p_t0), list(p_b0), list(p_b0), list(p_r0), list(p_r0)
curr_t_heat, curr_b_heat, curr_r_heat = list(p_t0), list(p_b0), list(p_r0)
v_pixel_dist = max(1, abs(p_t0[1] - p_b0[1]))

# ---------------------------------------------------------
# [Window 2 & 4] Tkinter 기반 그래프 창 생성 (위치 고정)
# ---------------------------------------------------------
def create_graph_win(title, x, y):
    win = tk.Toplevel(root_main); win.title(title)
    win.geometry(f"{W_4}x{H_4}+{x}+{y}")
    fig = plt.Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return fig, canvas

fig2, canvas2 = create_graph_win("Window 2: Heatmap Energy", 0, H_4 + 40)
ax2 = fig2.add_subplot(111); line_heat, = ax2.plot([], [], 'tab:orange', lw=2)
ax2.set_title("Heatmap Energy Intensity"); ax2.grid(True, alpha=0.3)

fig4, canvas4 = create_graph_win("Window 4: Physical Metrics", W_4, H_4 + 40)
ax4 = fig4.add_subplot(111); line_cm, = ax4.plot([], [], 'tab:blue', label='Disp')
ax4_ang = ax4.twinx(); line_deg, = ax4_ang.plot([], [], 'tab:red', ls='--', label='Angle')
ax4.set_title(f"Disp(cm) & Angle(deg) - H:{final_height}m"); ax4.grid(True, alpha=0.3)

# OpenCV 상단 영상 창
cv2.namedWindow("Window 1: Heatmap Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Window 3: Ghost Tracking Video", cv2.WINDOW_NORMAL)
cv2.moveWindow("Window 1: Heatmap Video", 0, 0); cv2.resizeWindow("Window 1: Heatmap Video", W_4, H_4)
cv2.moveWindow("Window 3: Ghost Tracking Video", W_4, 0); cv2.resizeWindow("Window 3: Ghost Tracking Video", W_4, H_4)

h_cm, h_deg, h_intensity, t_hist = [], [], [], []
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

# ==========================================
# 3. 메인 분석 루프
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break
    gray_curr = preprocess(frame); rx, ry, rw, rh = roi_building

    # --- Engine 1 (Heatmap + 마커X) ---
    vis_heatmap = frame.copy(); mean_diff = 0
    try:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        _, warp_matrix = cv2.findTransformECC(gray_init[ry:ry+rh, rx:rx+rw], gray_curr[ry:ry+rh, rx:rx+rw], warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        diff = cv2.absdiff(gray_init[ry:ry+rh, rx:rx+rw], cv2.warpAffine(gray_curr[ry:ry+rh, rx:rx+rw], warp_matrix, (rw, rh), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
        mean_diff = np.mean(diff)
        heatmap = cv2.applyColorMap(cv2.normalize(cv2.GaussianBlur(diff, (15, 15), 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        vis_heatmap[ry:ry+rh, rx:rx+rw] = cv2.addWeighted(vis_heatmap[ry:ry+rh, rx:rx+rw], 1-HEATMAP_ALPHA, heatmap, HEATMAP_ALPHA, 0)
        curr_t_heat = track_heat_logic(p_t0, warp_matrix); curr_b_heat = track_heat_logic(p_b0, warp_matrix); curr_r_heat = track_heat_logic(p_r0, warp_matrix)
        for p, c in [(curr_t_heat, (255, 100, 0)), (curr_b_heat, (0, 255, 100)), (curr_r_heat, (0, 100, 255))]:
            cv2.drawMarker(vis_heatmap, (int(p[0]), int(p[1])), c, cv2.MARKER_TILTED_CROSS, MARKER_SIZE, 2)
    except: pass

    # --- Engine 2 (Ghost/Tracking + 마커+) ---
    vis_ghost = frame.copy()
    vis_ghost[ry:ry+rh, rx:rx+rw] = cv2.addWeighted(frame[ry:ry+rh, rx:rx+rw], 1-GHOST_ALPHA, first_frame[ry:ry+rh, rx:rx+rw], GHOST_ALPHA, 0)
    curr_t_ghost, _ = track_ghost_logic(gray_curr, tpl_top, curr_t_ghost, SEARCH_MARGIN)
    curr_b_ghost, _ = track_ghost_logic(gray_curr, tpl_bot, curr_b_ghost, SEARCH_MARGIN)
    curr_r_ghost, _ = track_ghost_logic(gray_curr, tpl_ref, curr_r_ghost, SEARCH_MARGIN)
    
    sx = curr_r_ghost[0]-p_r0[0]; rm = (curr_t_ghost[0]-p_t0[0]-sx)-(curr_b_ghost[0]-p_b0[0]-sx)
    val_cm = (rm / v_pixel_dist) * HEIGHT_CM; val_deg = np.degrees(np.arctan2(rm, v_pixel_dist))
    for p, c in [(curr_t_ghost, (255, 0, 0)), (curr_b_ghost, (0, 255, 0)), (curr_r_ghost, (0, 0, 255))]:
        cv2.drawMarker(vis_ghost, (int(p[0]), int(p[1])), c, cv2.MARKER_CROSS, MARKER_SIZE, 2)
    cv2.putText(vis_ghost, f"Disp: {val_cm:.2f}cm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 실시간 데이터 및 그래프 업데이트 ---
    now = datetime.now(); t_hist.append(now); h_cm.append(val_cm); h_deg.append(val_deg); h_intensity.append(mean_diff)
    if len(t_hist) > HISTORY_LIMIT: [h.pop(0) for h in [t_hist, h_cm, h_deg, h_intensity]]
    
    line_heat.set_data(t_hist, h_intensity); ax2.relim(); ax2.autoscale_view()
    line_cm.set_data(t_hist, h_cm); line_deg.set_data(t_hist, h_deg); ax4.relim(); ax4.autoscale_view(); ax4_ang.relim(); ax4_ang.autoscale_view()
    
    canvas2.draw(); canvas4.draw()
    root_main.update() # Tkinter 윈도우 매니저 갱신

    cv2.imshow("Window 1: Heatmap Video", vis_heatmap); cv2.imshow("Window 3: Ghost Tracking Video", vis_ghost)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()