import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import tkinter as tk
from tkinter import font as tkfont
import os

# ==========================================
# [CONFIG] 시스템 설정
# ==========================================
VIDEO_SOURCE = './destory_structure_video.mp4'
TEMPLATE_SIZE = 120    
SEARCH_MARGIN = 120    
CONFIDENCE_THRESHOLD = 0.55 
UPDATE_THRESHOLD = 0.90     
MARKER_SIZE = 20       
HEATMAP_ALPHA = 0.6    
GHOST_ALPHA = 0.4      
HISTORY_LIMIT = 200    

# 분석 로직 함수들 (이전과 동일)
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
    px, py = int(prev_pos[0]), int(prev_pos[1])
    th, tw = tpl.shape[:2]
    x1, y1 = max(0, px - tw//2 - margin), max(0, py - th//2 - margin)
    x2, y2 = min(img.shape[1], px + tw//2 + margin), min(img.shape[0], py + th//2 + margin)
    search_area = img[y1:y2, x1:x2]
    if search_area.shape[0] < th or search_area.shape[1] < tw: return prev_pos, 0.0
    res = cv2.matchTemplate(search_area, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    sx, sy = get_subpixel_peak(res, max_loc)
    return (x1 + sx + tw/2, y1 + sy + th/2), max_val

def track_heat_logic(init_pos, warp_matrix):
    px, py = init_pos
    new_x = warp_matrix[0, 0] * px + warp_matrix[0, 1] * py + warp_matrix[0, 2]
    new_y = warp_matrix[1, 0] * px + warp_matrix[1, 1] * py + warp_matrix[1, 2]
    return (new_x, new_y)

# UI 및 ROI 선택 (창 사이즈 배려를 최소화함)
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
            cv2.setMouseCallback(win, lambda *args: None)
            return (min(roi_coords[0], roi_coords[2]), min(roi_coords[1], roi_coords[3]), abs(roi_coords[0]-roi_coords[2]), abs(roi_coords[1]-roi_coords[3]))
        if cv2.waitKey(1) & 0xFF == ord('q'): return None

# [추가] 높이 입력 UI 함수
final_height = 5.0
def get_ruler_picker_ui(screen_w, screen_h):
    global final_height
    root = tk.Tk()
    root.title("구조물 사이즈 설정")
    win_w, win_h = 400, 500
    root.geometry(f"{win_w}x{win_h}+{int(screen_w/2 - win_w/2)}+{int(screen_h/2 - win_h/2)}")
    root.configure(bg="white")
    tk.Label(root, text="구조물의 실제 높이(m) 입력", bg="white", font=("맑은 고딕", 14, "bold")).pack(pady=30)
    val_label = tk.Label(root, text=f"{final_height:.1f}", bg="white", font=("Arial", 45, "bold"))
    val_label.pack()
    def update_val(v):
        global final_height
        final_height = float(v)
        val_label.config(text=f"{final_height:.1f}")
    scale = tk.Scale(root, from_=0.1, to=100.0, orient="horizontal", resolution=0.1, showvalue=False, length=300, command=update_val)
    scale.set(final_height); scale.pack(pady=20)
    tk.Button(root, text="저장 후 분석 시작", bg="#4CAF50", fg="white", font=("맑은 고딕", 12, "bold"), width=20, height=2, command=root.destroy).pack(pady=40)
    root.mainloop()
    return final_height


# ==========================================
# 초기화 및 실행
# ==========================================
# [중요] 이전 실행 잔상 제거
plt.close('all') 
cv2.destroyAllWindows()

cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, first_frame = cap.read()
if not ret: exit()

root_info = tk.Tk(); SW, SH = root_info.winfo_screenwidth(), root_info.winfo_screenheight(); root_info.destroy()

# 셋업 창 (사이즈 강제 조절 제거, WINDOW_NORMAL로 설정)
win_setup = "DIC Setup"
cv2.namedWindow(win_setup, cv2.WINDOW_NORMAL) 
roi_building = get_roi_mouse(win_setup, first_frame)
roi_fixed = get_roi_mouse(win_setup, first_frame)
cv2.destroyWindow(win_setup)

BUILDING_HEIGHT_M = get_ruler_picker_ui(SW, SH)
HEIGHT_CM = BUILDING_HEIGHT_M * 100

# ROI 기반 초기 데이터 설정
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
def preprocess(img): return clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
gray_init = preprocess(first_frame)

# 특징점 및 템플릿 생성
def find_feat(img, roi):
    x, y, w, h = roi; c = cv2.goodFeaturesToTrack(img[y:y+h, x:x+w], 1, 0.01, 10)
    return (int(x + c[0][0][0]), int(y + c[0][0][1])) if c is not None else (x+w//2, y+h//2)

tx, ty = find_feat(gray_init, (roi_building[0], roi_building[1], roi_building[2], int(roi_building[3]*0.2)))
bx, by = find_feat(gray_init, (roi_building[0], roi_building[1] + int(roi_building[3]*0.8), roi_building[2], int(roi_building[3]*0.2)))
rx, ry = find_feat(gray_init, roi_fixed)

def get_safe_tpl(img, x, y):
    x1, y1 = max(0, x-TEMPLATE_SIZE//2), max(0, y-TEMPLATE_SIZE//2)
    x2, y2 = min(img.shape[1], x1+TEMPLATE_SIZE), min(img.shape[0], y1+TEMPLATE_SIZE)
    return img[y1:y2, x1:x2].copy(), (x1+TEMPLATE_SIZE//2, y1+TEMPLATE_SIZE//2)

tpl_top, p_t0 = get_safe_tpl(gray_init, tx, ty)
tpl_bot, p_b0 = get_safe_tpl(gray_init, bx, by)
tpl_ref, p_r0 = get_safe_tpl(gray_init, rx, ry)

# 좌표 변수 분리
curr_t_ghost, prev_t_ghost, curr_b_ghost, prev_b_ghost, curr_r_ghost, prev_r_ghost = list(p_t0), list(p_t0), list(p_b0), list(p_b0), list(p_r0), list(p_r0)
curr_t_heat, curr_b_heat, curr_r_heat = list(p_t0), list(p_b0), list(p_r0)

# [창 4개 설정 - 불필요한 Figure 1 방지]
plt.ion()
# Window 2: 히트맵 강도
fig_heat = plt.figure("Window 2: Heatmap Energy", figsize=(6, 4))
ax_heat = fig_heat.add_subplot(111); line_heat, = ax_heat.plot([], [], 'tab:orange', lw=2)
# Window 4: 변위 및 각도
fig_phys = plt.figure("Window 4: Physical Metrics", figsize=(6, 4))
ax_phys = fig_phys.add_subplot(111); line_cm, = ax_phys.plot([], [], 'tab:blue'); ax_ang = ax_phys.twinx(); line_deg, = ax_ang.plot([], [], 'tab:red', ls='--')

h_cm, h_deg, h_intensity, t_hist = [], [], [], []
# Window 1 & 3: 영상 창 (사용자가 크기 조절 가능하게)
cv2.namedWindow("Window 1: Heatmap Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Window 3: Ghost Tracking Video", cv2.WINDOW_NORMAL)

# ---------------------------------------------------------
# 루프 시작
# ---------------------------------------------------------
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

while True:
    ret, frame = cap.read()
    if not ret: break
    gray_curr = preprocess(frame)
    rx, ry, rw, rh = roi_building

    # --- [Engine 1: Heatmap & ECC] ---
    vis_heatmap = frame.copy(); mean_diff = 0
    try:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        ref_roi_g = gray_init[ry:ry+rh, rx:rx+rw]; curr_roi_g = gray_curr[ry:ry+rh, rx:rx+rw]
        _, warp_matrix = cv2.findTransformECC(ref_roi_g, curr_roi_g, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        aligned_roi = cv2.warpAffine(curr_roi_g, warp_matrix, (rw, rh), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        diff = cv2.absdiff(ref_roi_g, aligned_roi); mean_diff = np.mean(diff)
        norm_diff = cv2.normalize(cv2.GaussianBlur(diff, (15, 15), 0), None, 0, 255, cv2.NORM_MINMAX)
        vis_heatmap[ry:ry+rh, rx:rx+rw] = cv2.addWeighted(vis_heatmap[ry:ry+rh, rx:rx+rw], 1-HEATMAP_ALPHA, cv2.applyColorMap(norm_diff.astype(np.uint8), cv2.COLORMAP_JET), HEATMAP_ALPHA, 0)
        curr_t_heat = track_heat_logic(p_t0, warp_matrix); curr_b_heat = track_heat_logic(p_b0, warp_matrix); curr_r_heat = track_heat_logic(p_r0, warp_matrix)
    except: pass
    for p, c in [(curr_t_heat, (255, 100, 0)), (curr_b_heat, (0, 255, 100)), (curr_r_heat, (0, 100, 255))]:
        cv2.drawMarker(vis_heatmap, (int(p[0]), int(p[1])), c, cv2.MARKER_TILTED_CROSS, MARKER_SIZE, 2)

    # --- [Engine 2: Ghost & Matching] ---
    vis_ghost = frame.copy()
    vis_ghost[ry:ry+rh, rx:rx+rw] = cv2.addWeighted(frame[ry:ry+rh, rx:rx+rw], 1-GHOST_ALPHA, first_frame[ry:ry+rh, rx:rx+rw], GHOST_ALPHA, 0)
    def update_ghost(tpl, curr, prev):
        pos, conf = track_ghost_logic(gray_curr, tpl, curr, SEARCH_MARGIN)
        if conf < CONFIDENCE_THRESHOLD: return [curr[0]+(curr[0]-prev[0]), curr[1]+(curr[1]-prev[1])], curr
        return list(pos), curr
    curr_t_ghost, prev_t_ghost = update_ghost(tpl_top, curr_t_ghost, prev_t_ghost)
    curr_b_ghost, prev_b_ghost = update_ghost(tpl_bot, curr_b_ghost, prev_b_ghost)
    curr_r_ghost, prev_r_ghost = update_ghost(tpl_ref, curr_r_ghost, prev_r_ghost)
    
    sx = curr_r_ghost[0]-p_r0[0]; rm = (curr_t_ghost[0]-p_t0[0]-sx)-(curr_b_ghost[0]-p_b0[0]-sx)
    val_cm = (rm / max(1, abs(p_t0[1]-p_b0[1]))) * HEIGHT_CM
    val_deg = np.degrees(np.arctan2(rm, abs(p_t0[1]-p_b0[1])))
    
    for p, c in [(curr_t_ghost, (255, 0, 0)), (curr_b_ghost, (0, 255, 0)), (curr_r_ghost, (0, 0, 255))]:
        cv2.drawMarker(vis_ghost, (int(p[0]), int(p[1])), c, cv2.MARKER_CROSS, MARKER_SIZE, 2)

    # --- [Data & Graphs] ---
    now = datetime.now(); t_hist.append(now); h_cm.append(val_cm); h_deg.append(val_deg); h_intensity.append(mean_diff)
    if len(t_hist) > HISTORY_LIMIT: [h.pop(0) for h in [t_hist, h_cm, h_deg, h_intensity]]
    line_heat.set_data(t_hist, h_intensity); ax_heat.relim(); ax_heat.autoscale_view()
    line_cm.set_data(t_hist, h_cm); line_deg.set_data(t_hist, h_deg); ax_phys.relim(); ax_phys.autoscale_view(); ax_ang.relim(); ax_ang.autoscale_view()
    plt.pause(0.001)

    cv2.imshow("Window 1: Heatmap Video", vis_heatmap); cv2.imshow("Window 3: Ghost Tracking Video", vis_ghost)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()