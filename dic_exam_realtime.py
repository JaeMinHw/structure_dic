import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ==========================================
# 1. 설정값 및 광학 보정 세팅
# ==========================================
VIDEO_SOURCE = 0            # 0은 웹캠, 혹은 영상 파일 경로
BUILDING_HEIGHT_M = 0.3    # 추적할 상/하단 사이의 실제 수직 거리 (m)
HEIGHT_CM = BUILDING_HEIGHT_M * 100

# [광학 설정] 그림자 및 조명 변화 방지를 위한 CLAHE 필터
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 가우시안 블러로 미세한 광학 노이즈 제거 후 CLAHE 적용
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return clahe.apply(blurred)

def find_best_feature(gray_img, roi):
    x, y, w, h = roi
    sub_img = gray_img[y:y+h, x:x+w]
    if sub_img.size == 0: return x + w//2, y + h//2
    # 해당 영역 내에서 가장 추적하기 좋은 '코너'점을 자동으로 찾음
    corners = cv2.goodFeaturesToTrack(sub_img, maxCorners=1, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        cx, cy = corners[0][0]
        return int(x + cx), int(y + cy)
    return x + w//2, y + h//2

# ==========================================
# 2. 초기화 및 스마트 영역 설정 (2번의 드래그)
# ==========================================
cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, first_frame = cap.read()
if not ret: exit()

gray_init = preprocess(first_frame)
H_img, W_img = gray_init.shape

print("\n[스마트 실시간 모니터링]")
print("1. 건물 전체를 드래그하세요 (상/하 특징점 자동 추출)")
roi_b = cv2.selectROI("Step 1: Building Area", first_frame, False)
print("2. 고정된 배경(바닥/바위)을 드래그하세요 (카메라 흔들림 보정용)")
roi_r = cv2.selectROI("Step 2: Fixed Reference", first_frame, False)
cv2.destroyAllWindows()

# 상단/하단/기준점 자동 탐색
tx, ty = find_best_feature(gray_init, (roi_b[0], roi_b[1], roi_b[2], int(roi_b[3]*0.2)))
bx, by = find_best_feature(gray_init, (roi_b[0], roi_b[1] + int(roi_b[3]*0.8), roi_b[2], int(roi_b[3]*0.2)))
rx, ry = find_best_feature(gray_init, roi_r)

T_W, T_H = 60, 60
def get_safe_tpl(img, x, y):
    x1, y1 = max(0, x - T_W//2), max(0, y - T_H//2)
    x2, y2 = min(W_img, x1 + T_W), min(H_img, y1 + T_H)
    return img[y1:y2, x1:x2], (x1 + T_W//2, y1 + T_H//2)

tpl_t, p_t0 = get_safe_tpl(gray_init, tx, ty)
tpl_b, p_b0 = get_safe_tpl(gray_init, bx, by)
tpl_r, p_r0 = get_safe_tpl(gray_init, rx, ry)

v_pixel_dist = abs(p_t0[1] - p_b0[1]) # 비례식 기준이 되는 픽셀 높이

# ==========================================
# 3. 실시간 현재 시각 그래프 설정
# ==========================================
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot([], [], 'b-', linewidth=1.5, label='Tilt Displacement (cm)')
ax.set_title("Building Tilt Monitor (Real-time Clock)", fontsize=14)
ax.set_ylabel("Displacement (cm)", fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
value_display = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='red', fontweight='bold')

history_cm = []
history_time = []

# ==========================================
# 4. 분석 루프
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break
    gray_frame = preprocess(frame)

    def track(img, tpl):
        res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        return max_loc[0] + tpl.shape[1]//2, max_loc[1] + tpl.shape[0]//2

    curr_t = track(gray_frame, tpl_t)
    curr_b = track(gray_frame, tpl_b)
    curr_r = track(gray_frame, tpl_r)

    # 변위 계산 및 카메라 흔들림 제거
    shake_x = curr_r[0] - p_r0[0]
    net_t = (curr_t[0] - p_t0[0]) - shake_x
    net_b = (curr_b[0] - p_b0[0]) - shake_x
    
    # 최종 cm 변위 계산 (높이 비례식 방식)
    # v_pixel_dist(화면높이) : HEIGHT_CM(실제높이) = net_t-net_b(화면변위) : X(실제변위)
    tilt_cm = ((net_t - net_b) / v_pixel_dist) * HEIGHT_CM
    
    now = datetime.now()
    history_cm.append(tilt_cm)
    history_time.append(now)
    if len(history_cm) > 200:
        history_cm.pop(0)
        history_time.pop(0)

    # 그래프 업데이트
    line.set_data(history_time, history_cm)
    value_display.set_text(f"Current: {tilt_cm:.3f} cm")
    ax.relim(); ax.autoscale_view(); plt.pause(0.001)

    # 시각화 (화면 표시)
    for p, col, txt in [(curr_t, (255,0,0), "TOP"), (curr_b, (0,255,0), "BOT"), (curr_r, (0,0,255), "REF")]:
        cv2.drawMarker(frame, p, col, cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, txt, (p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    cv2.putText(frame, f"Disp: {tilt_cm:.2f} cm | Time: {now.strftime('%H:%M:%S')}", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
    cv2.imshow("Smart Real-time DIC", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()