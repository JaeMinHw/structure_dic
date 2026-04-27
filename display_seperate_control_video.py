import matplotlib
matplotlib.use('TkAgg')
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
# VIDEO_SOURCE = './destory_structure_video.mp4'
# VIDEO_SOURCE = './test_video_1.mp4'
VIDEO_SOURCE = './stop_video2.mp4'
TEMPLATE_SIZE = 120    
SEARCH_MARGIN = 100
CONFIDENCE_THRESHOLD = 0.55  
UPDATE_THRESHOLD = 0.90     
MARKER_SIZE = 20       
HEATMAP_ALPHA = 0.6    
GHOST_ALPHA = 0.4      
HISTORY_LIMIT = 500    

# 상태 변수
is_playing = True
current_frame_idx = 0
marker_history = []  

# ==========================================
# 1. 분석 로직 함수들
# ==========================================
def get_subpixel_peak(res, loc):
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

def track_ghost_logic(img, tpl, prev_pos, margin):
    px, py = int(prev_pos[0]), int(prev_pos[1]); th, tw = tpl.shape[:2]
    x1, y1 = max(0, px - tw//2 - margin), max(0, py - th//2 - margin)
    x2, y2 = min(img.shape[1], px + tw//2 + margin), min(img.shape[0], py + th//2 + margin)
    search_area = img[y1:y2, x1:x2]
    if search_area.shape[0] < th or search_area.shape[1] < tw: return prev_pos, 0.0
    res = cv2.matchTemplate(search_area, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    sx, sy = get_subpixel_peak(res, max_loc)
    return (x1 + sx + tw/2, y1 + sy + th/2), max_val

# ==========================================
# 2. UI 및 ROI 선택
# ==========================================
cv2.destroyAllWindows()
root_main = tk.Tk()
SW, SH = root_main.winfo_screenwidth(), root_main.winfo_screenheight()
root_main.withdraw()

W_4, H_4 = int(SW / 2), int(SH / 2) - 100

cap = cv2.VideoCapture(VIDEO_SOURCE); ret, first_frame = cap.read()
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if not ret: exit()

# ---------------------------------------------------------
# [수정 및 추가된 부분 1] 영상 원본 비율 계산
# ---------------------------------------------------------
vw = first_frame.shape[1] # 원본 영상 가로 해상도
vh = first_frame.shape[0] # 원본 영상 세로 해상도

# 메인 창에 띄울 크기 계산 (모니터 절반 크기에 맞춤)
scale_main = min(W_4 / vw, H_4 / vh)
disp_w = int(vw * scale_main)
disp_h = int(vh * scale_main)

# 셋업 창에 띄울 크기 계산 (모니터 70% 크기에 맞춤)
scale_setup = min((SW * 0.7) / vw, (SH * 0.7) / vh)
setup_w = int(vw * scale_setup)
setup_h = int(vh * scale_setup)
# ---------------------------------------------------------

win_setup = "DIC Setup"
cv2.namedWindow(win_setup, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

# [수정 및 추가된 부분 2] 셋업 창 크기 적용
cv2.resizeWindow(win_setup, setup_w, setup_h)

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

roi_building = get_roi_mouse(win_setup, first_frame)
roi_fixed = get_roi_mouse(win_setup, first_frame)
cv2.destroyWindow(win_setup)

final_height = 5.0
def get_ruler_picker_ui():
    global final_height
    h_win = tk.Toplevel(root_main); h_win.title("구조물 높이 설정")
    h_win.geometry(f"400x500+{(SW-400)//2}+{(SH-500)//2}")
    v_lbl = tk.Label(h_win, text=f"{final_height:.1f}", font=("Arial", 45, "bold")); v_lbl.pack(pady=30)
    def update_val(v): global final_height; final_height = float(v); v_lbl.config(text=f"{final_height:.1f}")
    scale = tk.Scale(h_win, from_=0.1, to=100.0, orient="horizontal", resolution=0.1, length=300, showvalue=False, command=update_val)
    scale.set(final_height); scale.pack(pady=20)
    tk.Button(h_win, text="저장", width=20, command=h_win.destroy).pack(pady=40)
    h_win.grab_set(); root_main.wait_window(h_win)

get_ruler_picker_ui(); HEIGHT_CM = final_height * 100
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
curr_t_ghost, prev_t_ghost = list(p_t0), list(p_t0)
curr_b_ghost, prev_b_ghost = list(p_b0), list(p_b0)
curr_r_ghost, prev_r_ghost = list(p_r0), list(p_r0)
v_pixel_dist = max(1, abs(p_t0[1] - p_b0[1]))

# ---------------------------------------------------------
# [재생 제어 및 그래프]
# ---------------------------------------------------------
h_cm, h_deg, h_intensity, t_hist = [], [], [], []

def on_slider_move(val):
    global current_frame_idx, is_playing, t_hist, h_cm, h_deg, h_intensity, marker_history
    global curr_t_ghost, prev_t_ghost, curr_b_ghost, prev_b_ghost, curr_r_ghost, prev_r_ghost

    new_idx = int(val)
    current_frame_idx = new_idx
    
    if t_hist:
        split_idx = len(t_hist)
        for i, f_idx in enumerate(t_hist):
            if f_idx > new_idx:
                split_idx = i
                break
        
        t_hist = t_hist[:split_idx]
        h_cm = h_cm[:split_idx]
        h_deg = h_deg[:split_idx]
        h_intensity = h_intensity[:split_idx]
        marker_history = marker_history[:split_idx]

        if len(marker_history) >= 1:
            last_pos = marker_history[-1]
            curr_t_ghost, curr_b_ghost, curr_r_ghost = list(last_pos[0]), list(last_pos[1]), list(last_pos[2])
            if len(marker_history) >= 2:
                prev_pos = marker_history[-2]
                prev_t_ghost, prev_b_ghost, prev_r_ghost = list(prev_pos[0]), list(prev_pos[1]), list(prev_pos[2])
        else:
            curr_t_ghost, prev_t_ghost = list(p_t0), list(p_t0)
            curr_b_ghost, prev_b_ghost = list(p_b0), list(p_b0)
            curr_r_ghost, prev_r_ghost = list(p_r0), list(p_r0)

    if not is_playing:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

def create_graph_win(title, x, y):
    win = tk.Toplevel(root_main); win.title(title)
    win.geometry(f"{W_4}x{H_4}+{x}+{y}")
    fig = plt.Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return fig, canvas

def resize_and_pad(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    # 검은색 캔버스 생성
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas



fig2, canvas2 = create_graph_win("Window 2: Heatmap Energy", 0, H_4 + 80)
ax2 = fig2.add_subplot(111); line_heat, = ax2.plot([], [], 'tab:orange', lw=2)
fig4, canvas4 = create_graph_win("Window 4: Physical Metrics", W_4, H_4 + 80)
ax4 = fig4.add_subplot(111); line_cm, = ax4.plot([], [], 'tab:blue'); ax4_ang = ax4.twinx(); line_deg, = ax4_ang.plot([], [], 'tab:red', ls='--')

ctrl_win = tk.Toplevel(root_main); ctrl_win.geometry(f"{SW}x80+0+{SH-150}")
btn_play = tk.Button(ctrl_win, text="PAUSE", width=10, command=lambda: toggle_play())
def toggle_play(): global is_playing; is_playing = not is_playing; btn_play.config(text="PAUSE" if is_playing else "PLAY")
btn_play.pack(side=tk.LEFT, padx=20)
frame_slider = tk.Scale(ctrl_win, from_=0, to=total_frames-1, orient=tk.HORIZONTAL, length=SW-200, command=on_slider_move)
frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

# ---------------------------------------------------------
# [수정된 부분] 메인 화면 위치 및 1/4 고정 크기 적용
# ---------------------------------------------------------
cv2.namedWindow("Window 1: Heatmap Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Window 3: Ghost Tracking Video", cv2.WINDOW_NORMAL)

cv2.moveWindow("Window 1: Heatmap Video", 0, 0)
cv2.resizeWindow("Window 1: Heatmap Video", W_4, H_4)

cv2.moveWindow("Window 3: Ghost Tracking Video", W_4, 0)
cv2.resizeWindow("Window 3: Ghost Tracking Video", W_4, H_4)
# ---------------------------------------------------------

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

# ==========================================
# 3. 메인 분석 루프
# ==========================================
last_good_warp = np.eye(2, 3, dtype=np.float32)
while True:
    if is_playing:
        ret, frame = cap.read()
        if not ret: is_playing = False; continue
        current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_slider.set(current_frame_idx)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        if not ret: root_main.update(); continue

    gray_curr = preprocess(frame); rx, ry, rw, rh = roi_building

    # --- Engine 1 (Heatmap) ---
    vis_heatmap = frame.copy(); mean_diff = 0
    try:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        _, warp_matrix = cv2.findTransformECC(gray_init[ry:ry+rh, rx:rx+rw], gray_curr[ry:ry+rh, rx:rx+rw], warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        
        # 성공하면 마지막 성공 매트릭스 업데이트
        last_good_warp = warp_matrix.copy() 
        
        diff = cv2.absdiff(gray_init[ry:ry+rh, rx:rx+rw], cv2.warpAffine(gray_curr[ry:ry+rh, rx:rx+rw], warp_matrix, (rw, rh), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
        mean_diff = np.mean(diff)
        heatmap = cv2.applyColorMap(cv2.normalize(cv2.GaussianBlur(diff, (15, 15), 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        vis_heatmap[ry:ry+rh, rx:rx+rw] = cv2.addWeighted(vis_heatmap[ry:ry+rh, rx:rx+rw], 1-HEATMAP_ALPHA, heatmap, HEATMAP_ALPHA, 0)
    
    except: 
        # 실패 시 무시(pass)하지 않고, 마지막으로 성공했던 매트릭스로 억지로 정렬해서 히트맵을 그림
        aligned_roi = cv2.warpAffine(gray_curr[ry:ry+rh, rx:rx+rw], last_good_warp, (rw, rh), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        diff = cv2.absdiff(gray_init[ry:ry+rh, rx:rx+rw], aligned_roi)
        mean_diff = np.mean(diff)
        heatmap = cv2.applyColorMap(cv2.normalize(cv2.GaussianBlur(diff, (15, 15), 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        vis_heatmap[ry:ry+rh, rx:rx+rw] = cv2.addWeighted(vis_heatmap[ry:ry+rh, rx:rx+rw], 1-HEATMAP_ALPHA, heatmap, HEATMAP_ALPHA, 0)

    # --- Engine 2 (Ghost/Tracking) ---
    vis_ghost = frame.copy()
    vis_ghost[ry:ry+rh, rx:rx+rw] = cv2.addWeighted(frame[ry:ry+rh, rx:rx+rw], 1-GHOST_ALPHA, first_frame[ry:ry+rh, rx:rx+rw], GHOST_ALPHA, 0)
    
    def update_ghost(tpl, curr, prev):
        pos, conf = track_ghost_logic(gray_curr, tpl, curr, SEARCH_MARGIN)
        # conf가 낮아도 무조건 화면에서 찾은 위치(pos)를 믿고 따라감
        return list(pos), curr

    curr_t_ghost, prev_t_ghost = update_ghost(tpl_top, curr_t_ghost, prev_t_ghost)
    curr_b_ghost, prev_b_ghost = update_ghost(tpl_bot, curr_b_ghost, prev_b_ghost)
    curr_r_ghost, prev_r_ghost = update_ghost(tpl_ref, curr_r_ghost, prev_r_ghost)
    
    sx = curr_r_ghost[0]-p_r0[0]; rm = (curr_t_ghost[0]-p_t0[0]-sx)-(curr_b_ghost[0]-p_b0[0]-sx)
    val_cm = (rm / v_pixel_dist) * HEIGHT_CM; val_deg = np.degrees(np.arctan2(rm, v_pixel_dist))
    
    # --- [중요] 매 프레임 위치를 히스토리에 기록 ---
    if is_playing:
        marker_history.append((tuple(curr_t_ghost), tuple(curr_b_ghost), tuple(curr_r_ghost)))
        t_hist.append(current_frame_idx)
        h_cm.append(val_cm); h_deg.append(val_deg); h_intensity.append(mean_diff)
        if len(t_hist) > HISTORY_LIMIT: 
            [h.pop(0) for h in [t_hist, h_cm, h_deg, h_intensity, marker_history]]

    # 시각화 및 그래프 업데이트
    for p, c in [(curr_t_ghost, (255, 0, 0)), (curr_b_ghost, (0, 255, 0)), (curr_r_ghost, (0, 0, 255))]:
        cv2.drawMarker(vis_ghost, (int(p[0]), int(p[1])), c, cv2.MARKER_CROSS, MARKER_SIZE, 2)
    
    line_heat.set_data(t_hist, h_intensity); ax2.relim(); ax2.autoscale_view()
    line_cm.set_data(t_hist, h_cm); line_deg.set_data(t_hist, h_deg); ax4.relim(); ax4.autoscale_view(); ax4_ang.relim(); ax4_ang.autoscale_view()
    canvas2.draw(); canvas4.draw()
    
    # (기존 시각화 및 그래프 업데이트 코드 ...)
    line_heat.set_data(t_hist, h_intensity); ax2.relim(); ax2.autoscale_view()
    line_cm.set_data(t_hist, h_cm); line_deg.set_data(t_hist, h_deg); ax4.relim(); ax4.autoscale_view(); ax4_ang.relim(); ax4_ang.autoscale_view()
    canvas2.draw(); canvas4.draw()
    
    # [새로 추가된 부분] W_4, H_4 사이즈의 검은 여백 캔버스에 영상 합성
    disp_heatmap = resize_and_pad(vis_heatmap, W_4, H_4)
    disp_ghost = resize_and_pad(vis_ghost, W_4, H_4)
    
    # [수정된 부분] 여백이 추가된 영상을 띄움
    cv2.imshow("Window 1: Heatmap Video", disp_heatmap)
    cv2.imshow("Window 3: Ghost Tracking Video", disp_ghost)
    
    root_main.update()
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()