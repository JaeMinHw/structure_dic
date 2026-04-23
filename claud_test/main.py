"""
건물 기울어짐 감시 시스템 - 메인 실행 파일

[저장 영상 모드]
  python main.py --source video.mp4 --building-height 20.0

[실시간 카메라 모드]
  python main.py --source 0 --building-height 20.0

키 조작
  SPACE : 일시정지 / 재개
  q     : 종료
  r     : 추적기 재초기화
  ←/→   : 영상 5초 뒤로/앞으로 (영상 파일 모드)
"""

import argparse
import os
import cv2
import time

import numpy as np
from tracker import HybridTracker
from compensator import MotionCompensator
from converter import PhysicsConverter
from alerter import Alerter, AlertState
from visualizer import Visualizer
from logger import DataLogger


def parse_args():
    p = argparse.ArgumentParser(description="건물 기울어짐 감시 시스템")
    p.add_argument("--source",          default="C:/Users/admin_user/Desktop/dic_test2.mp4",
                   help="영상 파일 경로 또는 카메라 번호 (예: video.mp4 / 0)")
    p.add_argument("--building-height", type=float, default=20.0,
                   help="건물 실제 높이 (m)")
    p.add_argument("--pixel-height",    type=int,   default=0,
                   help="건물 픽셀 높이 (0=ROI 기반 자동 계산)")
    p.add_argument("--alert-cm",        type=float, default=3.0,
                   help="경보 임계값 (cm)")
    p.add_argument("--alert-deg",       type=float, default=0.5,
                   help="경보 각도 임계값 (deg)")
    p.add_argument("--log-dir",         default="logs",
                   help="로그 저장 폴더")
    p.add_argument("--speed",           type=float, default=1.0,
                   help="재생 속도 배율 (0.5=절반, 2.0=2배, 영상 모드 전용)")
    p.add_argument("--no-display",      action="store_true",
                   help="화면 출력 비활성화 (배치 처리용)")
    return p.parse_args()


def _is_video_file(source: str) -> bool:
    """경로가 영상 파일이면 True, 카메라 번호면 False"""
    return not str(source).isdigit()


def _open_capture(source: str) -> cv2.VideoCapture:
    if _is_video_file(source):
        if not os.path.exists(source):
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {source}")
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(f"소스를 열 수 없습니다: {source}")
    return cap


def _select_rois(frame: "np.ndarray") -> list:
    """첫 프레임에서 3개 ROI를 드래그로 지정"""
    prompts = [
        "[1/3] 상단 구역 드래그 → ENTER",
        "[2/3] 하단 구역 드래그 → ENTER",
        "[3/3] 고정 참조점(배경) 드래그 → ENTER",
    ]
    rois = []
    
    # ROI 설정 창 크기를 70% 비율로 축소 설정
    h, w = frame.shape[:2]
    cv2.namedWindow("ROI 설정", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI 설정", int(w * 0.7), int(h * 0.7))

    for prompt in prompts:
        clone = frame.copy()
        cv2.putText(clone, prompt, (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        roi = cv2.selectROI("ROI 설정", clone, fromCenter=False, showCrosshair=True)
        if roi == (0, 0, 0, 0):
            cv2.destroyWindow("ROI 설정")
            raise ValueError(f"ROI 설정이 취소되었습니다: {prompt}")
        rois.append(roi)   # (x, y, w, h)
        
    cv2.destroyWindow("ROI 설정")
    return rois            # [top, bot, ref]


def _estimate_pixel_height(rois: list) -> int:
    top_roi, bot_roi, _ = rois
    top_cy = top_roi[1] + top_roi[3] // 2
    bot_cy = bot_roi[1] + bot_roi[3] // 2
    return max(abs(bot_cy - top_cy), 1)


def main():
    args = parse_args()
    is_video = _is_video_file(args.source)

    # ── 캡처 열기 ────────────────────────────────────────────────────────
    cap = _open_capture(args.source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else -1
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms     = max(1, int(1000 / (video_fps * args.speed))) if is_video else 1

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("첫 프레임을 읽을 수 없습니다.")

    h, w = first_frame.shape[:2]
    print(f"[INFO] 소스: {args.source}  해상도: {w}x{h}", end="")
    if is_video:
        print(f"  총 프레임: {total_frames}  FPS: {video_fps:.1f}  재생속도: x{args.speed}")
    else:
        print()

    # ── ROI 설정 ─────────────────────────────────────────────────────────
    print("[INFO] ROI 설정 창이 열립니다.")
    rois = _select_rois(first_frame)

    # ── 모듈 초기화 ──────────────────────────────────────────────────────
    pixel_height = args.pixel_height or _estimate_pixel_height(rois)
    scale        = PhysicsConverter.calc_scale(args.building_height, pixel_height)

    tracker     = HybridTracker(rois)
    compensator = MotionCompensator()
    converter   = PhysicsConverter(scale, args.building_height)
    alerter     = Alerter(args.alert_cm, args.alert_deg)
    visualizer  = Visualizer(w, h)
    logger      = DataLogger(args.log_dir)

    print(f"[INFO] 스케일: {scale:.4f} cm/pixel | 경보: {args.alert_cm}cm / {args.alert_deg}deg")
    print("[INFO] 시작. SPACE=일시정지  q=종료  r=재초기화", end="")
    if is_video:
        print("  ←/→=5초 이동")
    else:
        print()

    # ── 메인 창 크기 70% 설정 ───────────────────────────────────────────────
    if not args.no_display:
        cv2.namedWindow("Building Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Building Monitor", int(w * 0.7), int(h * 0.7))

    # ── 메인 루프 ────────────────────────────────────────────────────────
    frame_count = 0
    paused      = False
    fps_display = 0.0
    fps_timer   = time.time()
    result      = None
    alert       = AlertState()

    # 첫 프레임을 이미 읽었으므로 바로 처리
    pending_frame = first_frame

    while True:
        # ── 키 입력 처리 ──────────────────────────────────────────────
        key = cv2.waitKey(delay_ms if not paused else 50) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reinitialize(pending_frame if pending_frame is not None else frame)
            compensator.reset()
            print("[INFO] 추적기 재초기화 완료.")
        elif key == ord(' '):
            paused = not paused
            print(f"[INFO] {'일시정지' if paused else '재개'}")
        elif is_video and key == 81:   # ← 왼쪽 화살표
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - int(video_fps * 5)))
            pending_frame = None
        elif is_video and key == 83:   # → 오른쪽 화살표
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, cur + int(video_fps * 5)))
            pending_frame = None

        if paused:
            if not args.no_display and result is not None:
                cv2.imshow("Building Monitor", visualizer.draw(
                    pending_frame or frame, tracker, result, alert, fps_display))
            continue

        # ── 프레임 읽기 ───────────────────────────────────────────────
        if pending_frame is not None:
            frame = pending_frame
            pending_frame = None
        else:
            ret, frame = cap.read()
            if not ret:
                if is_video:
                    print("[INFO] 영상 재생 완료.")
                else:
                    print("[WARN] 프레임 읽기 실패. 재시도 중...")
                    time.sleep(0.05)
                    continue
                break

        frame_count += 1

        # ── 파이프라인 ────────────────────────────────────────────────
        raw_disp  = tracker.update(frame)
        comp_disp = compensator.compensate(raw_disp)
        result    = converter.convert(comp_disp)
        alert     = alerter.check(result)

        logger.log(frame_count, result, alert)

        # ── FPS 계산 ──────────────────────────────────────────────────
        if frame_count % 30 == 0:
            elapsed     = time.time() - fps_timer
            fps_display = 30.0 / max(elapsed, 1e-6)
            fps_timer   = time.time()

        # ── 화면 출력 ─────────────────────────────────────────────────
        if not args.no_display:
            display = visualizer.draw(frame, tracker, result, alert, fps_display)

            # 영상 진행 바 (영상 파일 모드만)
            if is_video and total_frames > 0:
                pos   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ratio = pos / total_frames
                bar_w = w - 20
                cv2.rectangle(display, (10, h - 14), (10 + bar_w, h - 6), (60, 60, 60), -1)
                cv2.rectangle(display, (10, h - 14), (10 + int(bar_w * ratio), h - 6),
                              (0, 200, 120), -1)
                time_sec = pos / video_fps
                tot_sec  = total_frames / video_fps
                time_str = f"{int(time_sec)//60:02d}:{int(time_sec)%60:02d} / {int(tot_sec)//60:02d}:{int(tot_sec)%60:02d}"
                cv2.putText(display, time_str, (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Building Monitor", display)

    cap.release()
    cv2.destroyAllWindows()
    logger.close()
    print(f"[INFO] 종료. 처리 프레임: {frame_count}")


if __name__ == "__main__":
    main()