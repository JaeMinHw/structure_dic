import cv2
import numpy as np

VIDEO_SOURCE = 'C:/Users/admin_user/Desktop/dic_test2.mp4'

# ==============================
# ROI 선택
# ==============================
roi_pts = []
selecting = False

def mouse_roi(event, x, y, flags, param):
    global roi_pts, selecting, temp_img, base_img

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_pts = [x, y]
        selecting = True

    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        temp_img = base_img.copy()
        cv2.rectangle(temp_img, (roi_pts[0], roi_pts[1]), (x, y), (0,255,0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_pts += [x, y]
        selecting = False


def get_roi(win, frame):
    global base_img, temp_img, roi_pts
    base_img = frame.copy()
    temp_img = frame.copy()
    roi_pts = []

    cv2.setMouseCallback(win, mouse_roi)

    while True:
        cv2.imshow(win, temp_img)
        if len(roi_pts) == 4:
            x1,y1,x2,y2 = roi_pts
            return (min(x1,x2), min(y1,y2), abs(x1-x2), abs(y1-y2))
        if cv2.waitKey(1) & 0xFF == 27:
            return None

# ==============================
# Edge 기반 직경 측정
# ==============================
def measure_pipe_diameter(gray, roi):
    x, y, w, h = roi
    sub = gray[y:y+h, x:x+w]

    edges = cv2.Canny(sub, 50, 150)

    diameters = []
    for i in range(5, h-5, max(1, h//20)):
        row = edges[i]
        xs = np.where(row > 0)[0]

        if len(xs) > 2:
            d = xs[-1] - xs[0]
            diameters.append(d)

    if len(diameters) == 0:
        return None

    return np.mean(diameters)

# ==============================
# 포인트 필터링 (핵심🔥)
# ==============================
def filter_points(pts, w, h, margin=10):
    if pts is None:
        return None

    pts = pts.reshape(-1,2)  # 🔥 강제 정리

    valid = []
    for x, y in pts:
        if margin < x < w-margin and margin < y < h-margin:
            valid.append([x, y])

    if len(valid) == 0:
        return None

    return np.array(valid, dtype=np.float32).reshape(-1,1,2)

# ==============================
# 특징점 초기화
# ==============================
def get_points(gray, roi, n=30):
    x,y,w,h = roi
    pts = cv2.goodFeaturesToTrack(gray[y:y+h, x:x+w], n, 0.01, 10)

    if pts is None:
        return None

    pts[:,0,0] += x
    pts[:,0,1] += y
    return pts.astype(np.float32)

# ==============================
# 초기 설정
# ==============================
cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, first = cap.read()
if not ret:
    print("영상 로드 실패")
    exit()

gray0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("setup", cv2.WINDOW_NORMAL)

print("1️⃣ 건물 ROI 선택")
roi_building = get_roi("setup", first)

print("2️⃣ 기준점 ROI 선택")
roi_ref = get_roi("setup", first)

print("3️⃣ 파이프 ROI 선택")
roi_pipe = get_roi("setup", first)

# ==============================
# 스케일 계산
# ==============================
pipe_px = measure_pipe_diameter(gray0, roi_pipe)

if pipe_px is None:
    print("파이프 검출 실패")
    exit()

print(f"파이프 픽셀 직경: {pipe_px:.2f}px")

real_diameter = float(input("파이프 실제 직경(mm): "))

mm_per_pixel = real_diameter / pipe_px

print(f"mm per pixel: {mm_per_pixel:.4f}")

# ==============================
# 특징점 초기화
# ==============================
pts_build = get_points(gray0, roi_building)
pts_ref = get_points(gray0, roi_ref)

if pts_build is None or pts_ref is None:
    print("특징점 부족")
    exit()

prev_gray = gray0.copy()

lk_params = dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# ==============================
# 메인 루프
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    nxt_b, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_build, None, **lk_params)
    nxt_r, st2, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_ref, None, **lk_params)

    if nxt_b is None or nxt_r is None:
        prev_gray = gray.copy()
        continue

    # status 필터링
    good_b = nxt_b[st1==1].reshape(-1,1,2)
    good_r = nxt_r[st2==1].reshape(-1,1,2)

    prev_b = pts_build[st1==1].reshape(-1,1,2)
    prev_r = pts_ref[st2==1].reshape(-1,1,2)

    # NaN 제거
    good_b = good_b[~np.isnan(good_b).any(axis=2)]
    good_r = good_r[~np.isnan(good_r).any(axis=2)]

    prev_b = prev_b[~np.isnan(prev_b).any(axis=2)]
    prev_r = prev_r[~np.isnan(prev_r).any(axis=2)]

    good_b = good_b.reshape(-1,1,2)
    good_r = good_r.reshape(-1,1,2)
    prev_b = prev_b.reshape(-1,1,2)
    prev_r = prev_r.reshape(-1,1,2)
    h, w = gray.shape

    # 🔥 경계 필터링 (핵심)
    good_b = filter_points(good_b, w, h)
    good_r = filter_points(good_r, w, h)

    if good_b is None or good_r is None:
        prev_gray = gray.copy()
        continue

    if len(good_b) < 5 or len(good_r) < 5:
        prev_gray = gray.copy()
        continue

    # 🔥 Subpixel refinement
    cv2.cornerSubPix(gray, good_b, (5,5), (-1,-1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    cv2.cornerSubPix(gray, good_r, (5,5), (-1,-1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    # 이동 계산
    move_b = np.mean(good_b - prev_b[:len(good_b)], axis=0)
    move_r = np.mean(good_r - prev_r[:len(good_r)], axis=0)

    real_move = move_b - move_r

    disp_mm = float(real_move[0][0]) * mm_per_pixel

    # 시각화
    for p in good_b:
        cv2.circle(frame, tuple(p[0].astype(int)), 2, (0,255,0), -1)

    for p in good_r:
        cv2.circle(frame, tuple(p[0].astype(int)), 2, (0,0,255), -1)

    cv2.putText(frame, f"Disp: {disp_mm:.2f} mm", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("result", frame)

    # 상태 업데이트
    prev_gray = gray.copy()
    pts_build = good_b.copy()
    pts_ref = good_r.copy()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()