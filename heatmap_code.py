import cv2
import numpy as np

def run_clean_dic_save(video_path, output_path='analysis_result.mp4'):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: 
        print("영상을 열 수 없습니다.")
        return

    # --- 영상 저장을 위한 설정 ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 코덱 설정
    fps = cap.get(cv2.CAP_PROP_FPS)         # 원본 영상의 FPS 가져오기
    if fps == 0: fps = 30.0                 # FPS 정보가 없을 경우 기본값 설정
    
    # 원본 해상도 (저장용 해상도)
    orig_h, orig_w = first_frame.shape[:2]
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
    # ---------------------------

    # 초기 설정 (DOI 선택)
    display_w, display_h = 960, 540
    show_img = cv2.resize(first_frame, (display_w, display_h))
    roi = cv2.selectROI('Select DOI', show_img, False)
    cv2.destroyWindow('Select DOI')
    
    if roi == (0, 0, 0, 0):
        print("영역이 선택되지 않았습니다.")
        return

    sx, sy = orig_w/display_w, orig_h/display_h
    x, y, w, h = int(roi[0]*sx), int(roi[1]*sy), int(roi[2]*sx), int(roi[3]*sy)

    # [중요] 기준점 저장 (첫 프레임)
    ref_roi_gray = cv2.cvtColor(first_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

    print(f"분석 및 저장을 시작합니다: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_frame = frame.copy()
        current_roi = current_frame[y:y+h, x:x+w]
        current_roi_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)

        # 1. 차이 계산 (기준 vs 현재)
        diff = cv2.absdiff(ref_roi_gray, current_roi_gray)

        # 2. 유령 제거 필터 (Threshold)
        # 40보다 작은 미세 변화(과거 잔상 등)는 무시
        _, diff_clean = cv2.threshold(diff, 40, 255, cv2.THRESH_TOZERO)

        # 3. 모폴로지 연산 (점 노이즈 제거)
        kernel = np.ones((5,5), np.uint8)
        diff_clean = cv2.morphologyEx(diff_clean, cv2.MORPH_OPEN, kernel)

        # 4. 히트맵 생성
        diff_blur = cv2.GaussianBlur(diff_clean, (15, 15), 0)
        norm_diff = cv2.normalize(diff_blur, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm_diff.astype(np.uint8), cv2.COLORMAP_JET)

        # 5. 현재 프레임 위에 히트맵 합성
        combined = cv2.addWeighted(current_roi, 0.4, heatmap, 0.6, 0)
        current_frame[y:y+h, x:x+w] = combined

        # 안내 문구 및 테두리 (저장될 영상에도 포함됨)
        cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(current_frame, "ANALYSIS SAVING...", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- 저장 및 출력 ---
        out.write(current_frame) # 가공된 현재 프레임을 파일에 기록
        
        cv2.imshow('Analysis & Saving', cv2.resize(current_frame, (display_w, display_h)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break


    # 리소스 해제
    cap.release()
    out.release() # 영상 저장 완료
    cv2.destroyAllWindows()
    print("저장이 완료되었습니다.")

# 실행
run_clean_dic_save('destory_structure_video.mp4', 'final_analysis.mp4')