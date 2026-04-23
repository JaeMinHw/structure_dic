import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# --- 1. SAM 모델 로드 및 설정 ---
def load_sam_model(model_path, device):
    print("SAM 모델 로딩 중...")
    try:
        sam = sam_model_registry["vit_b"](checkpoint=model_path)
        sam.to(device=device)
        print("SAM 모델 로딩 완료!")
        return SamPredictor(sam)
    except FileNotFoundError:
        print("모델 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None

# --- 2. 배경 인페인팅 함수 ---
def apply_inpainting(img, mask):
    kernel = np.ones((5,5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    inpainted = cv2.inpaint(img, dilated_mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def main():
    image_path = 'C:/Users/admin_user/Desktop/te.jpg' 
    model_path = 'sam_vit_b_01ec64.pth' 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = load_sam_model(model_path, device)
    if predictor is None: return

    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        return
    h, w = img.shape[:2]

    # --- [수정] 창 크기 결정을 위한 비율 계산 ---
    screen_w = 1000 # 원하는 가로 크기 (픽셀 단위)
    screen_h = int(h * (screen_w / w)) # 비율에 맞춘 세로 크기
    
    # 1. ROI 선택 창 설정
    roi_win_name = "Select Object Area (SAM)"
    cv2.namedWindow(roi_win_name, cv2.WINDOW_NORMAL) # 창 크기 조절 가능 모드
    cv2.resizeWindow(roi_win_name, screen_w, screen_h) # 계산한 크기로 설정
    # ------------------------------------------

    predictor.set_image(img)

    print("Step 1: 마우스로 객체 주위를 대강 드래그하세요. 완료 후 ENTER를 누르세요.")
    roi = cv2.selectROI(roi_win_name, img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(roi_win_name)

    x, y, w_box, h_box = roi
    if w_box == 0 or h_box == 0:
        print("영역이 선택되지 않았습니다.")
        return

    print("Step 2: AI 누끼 따는 중...")
    input_box = np.array([x, y, x + w_box, y + h_box])

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    best_mask = masks[0].astype(np.uint8) * 255 

    mask_roi = best_mask[y:y+h_box, x:x+w_box]
    obj_roi = img[y:y+h_box, x:x+w_box].copy()
    b, g, r = cv2.split(obj_roi)
    obj_rgba = cv2.merge([b, g, r, mask_roi])

    bg_base = apply_inpainting(img, best_mask)

    angle = 0
    print("Step 3: 도움말: [→] 오른쪽 1도, [←] 왼쪽 1도, [ESC] 종료")
    
    # --- [수정] 결과 확인 창 설정 ---
    rot_win_name = "Interactive Rotation"
    cv2.namedWindow(rot_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rot_win_name, screen_w, screen_h)
    # ------------------------------

    while True:
        M = cv2.getRotationMatrix2D((w_box//2, h_box//2), angle, 1.0)
        rotated_obj_rgba = cv2.warpAffine(obj_rgba, M, (w_box, h_box), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

        display_img = bg_base.copy()
        
        alpha = rotated_obj_rgba[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        bg_roi = display_img[y:y+h_box, x:x+w_box]

        for c in range(0, 3):
            bg_roi[:, :, c] = (alpha * rotated_obj_rgba[:, :, c] + alpha_inv * bg_roi[:, :, c]).astype(np.uint8)

        cv2.putText(display_img, f"Angle: {angle} deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow(rot_win_name, display_img)

        key = cv2.waitKeyEx(1)
        if key == 27: # ESC
            break
        elif key == 0x270000 or key == 83: # 오른쪽
            angle -= 1
        elif key == 0x250000 or key == 81: # 왼쪽
            angle += 1
        # --- [추가] 's' 키를 누르면 현재 이미지 저장 ---
        elif key == ord('s') or key == ord('S'): 
            save_path = f'rotated_result_{angle}deg.jpg'
            # 텍스트(Angle: ...)가 없는 깨끗한 이미지를 원하시면 
            # cv2.putText 코드 윗부분에서 save_img = display_img.copy()를 미리 해두세요.
            cv2.imwrite(save_path, display_img)
            print(f"이미지가 저장되었습니다: {save_path}")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()