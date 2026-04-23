import cv2
import numpy as np
from rembg import remove
from PIL import Image
import tkinter as tk
import io

# 1. 테스트할 이미지 경로 설정
# 본인의 컴퓨터에 있는 이미지 경로로 수정하세요.
input_path = 'C:/Users/admin_user/Desktop/te2.png' 
# ==========================================
# [마우스 핸들러] 드래그 앤 드롭 로직
# ==========================================
roi_coords, selecting = [], False
def mouse_drag_handler(event, x, y, flags, param):
    global roi_coords, selecting, temp_img, display_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_coords, selecting = [x, y], True
    
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        temp_img = display_img.copy()
        cv2.rectangle(temp_img, (roi_coords[0], roi_coords[1]), (x, y), (0, 255, 0), 2)
    
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        roi_coords.extend([x, y])

# ==========================================
# 메인 함수: 자동 화면 맞춤 누끼 추출
# ==========================================
def run_auto_fit_nooki(image_path):
    global temp_img, display_img, roi_coords, selecting
    
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 찾을 수 없습니다."); return

    # [핵심] 내 모니터 해상도 자동 감지
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    # 이미지가 모니터보다 크면 화면의 80% 크기로 자동 축소 비율 계산
    h, w = img.shape[:2]
    limit_w, limit_h = screen_w * 0.8, screen_h * 0.8
    scale = min(limit_w/w, limit_h/h, 1.0) 
    
    # [개선] 선명한 축소를 위해 INTER_AREA 보간법 사용
    display_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    temp_img = display_img.copy()
    roi_coords, selecting = [], False

    win_name = "Drag Area (Release to Nooki)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE) 
    cv2.setMouseCallback(win_name, mouse_drag_handler)

    print(f"모니터 크기 감지: {screen_w}x{screen_h}")
    print("마우스로 영역을 드래그하세요. 버튼을 떼면 바로 누끼를 땁니다.")

    while True:
        cv2.imshow(win_name, temp_img)
        if len(roi_coords) == 4: break
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            cv2.destroyAllWindows(); return

    cv2.destroyWindow(win_name)

    # 원본 좌표로 복원
    x1, y1, x2, y2 = [int(c / scale) for c in roi_coords]
    x, y, rw, rh = min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)

    # 누끼 작업
    crop = img[y:y+rh, x:x+rw]
    print("AI가 고화질로 누끼를 따고 있습니다...")
    nooki_rgba = remove(crop)

    # 결과물 출력 (결과창도 모니터 크기에 맞춰 자동 조절)
    res_h, res_w = nooki_rgba.shape[:2]
    res_scale = min(limit_w/res_w, limit_h/res_h, 1.0)
    
    rgb = nooki_rgba[:, :, :3]
    alpha = nooki_rgba[:, :, 3] / 255.0
    preview = (rgb * alpha[:, :, np.newaxis] + np.full_like(rgb, 200) * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)

    res_win = "Nooki Result"
    cv2.imshow(res_win, cv2.resize(preview, (int(res_w * res_scale), int(res_h * res_scale)), interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

run_auto_fit_nooki(input_path)