# 카메라 주소로 가서 영상 가져오는 코드

import cv2
from datetime import datetime

# 찾아내신 주소를 입력합니다.
stream_url = "http://169.254.113.19/live/169.254.113.19/jpeg/1/jpeg.php" 

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("에러: 스트림을 열 수 없습니다.")
    exit()

# ========================================================
# 💡 비디오 저장 설정 준비
# ========================================================
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f'record_{current_time}.avi' # 호환성이 좋은 avi 형식 사용

out = None
fps = 5.0 # 저장될 영상의 프레임 속도 (필요시 조절)

print(f"▶️ 영상 수신을 시작합니다. (종료하려면 영상 창을 클릭하고 'q'를 누르세요)")

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("영상을 가져올 수 없습니다. 스트림이 끊겼을 수 있습니다.")
            break

        # ========================================================
        # 💡 최초 1회: 영상 크기를 확인하고 VideoWriter 생성하기
        # ========================================================
        if out is None:
            height, width, _ = frame.shape 
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            out = cv2.VideoWriter(file_name, fourcc, fps, (width, height))
            print(f"✅ [{file_name}] 파일로 녹화가 시작되었습니다!")

        # 화면에 영상 띄우기 및 파일에 저장하기
        out.write(frame)
        cv2.imshow("Real-time Analysis & Recording", frame)

        # 'q' 키 입력 감지
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[정상 종료] 'q' 키를 눌렀습니다.")
            break

except KeyboardInterrupt:
    print("\n[강제 종료 감지] 터미널에서 강제로 중단했습니다.")
except Exception as e:
    print(f"\n[에러 발생] {e}")

finally:
    # ========================================================
    # 💡 자원 해제 (저장을 완료하고 창을 닫는 필수 과정)
    # ========================================================
    cap.release() # 카메라 연결 해제
    if out is not None:
        out.release() # 비디오 파일 안전하게 닫기
        print(f"💾 녹화 완료! [{file_name}] 파일이 성공적으로 저장되었습니다.")
    cv2.destroyAllWindows() # 열려있는 창 모두 닫기