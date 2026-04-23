import cv2
import numpy as np
import tkinter as tk
import sys
from pathlib import Path

# ============================================================
# 설정 영역
# ============================================================
IMG1_PATH = 'C:/Users/admin_user/Desktop/te.jpg'                    # 기준 이미지
IMG2_PATH = 'C:/Users/admin_user/Desktop/rotated_result_7deg.jpg'   # 변형 이미지

TEMPLATE_SIZE   = 100   # 템플릿 크기 (픽셀, 홀수 권장 아님 — 짝수도 무방)
SEARCH_MARGIN   = 100   # 템플릿 주변 탐색 여유 범위 (픽셀)
SUBPIXEL        = True  # 서브픽셀 정밀도 활성화
SCREEN_RATIO    = 0.88  # 모니터 대비 표시 창 비율
# ============================================================


# ──────────────────────────────────────────────────────────────
# 유틸 함수
# ──────────────────────────────────────────────────────────────
def get_screen_size() -> tuple[int, int]:
    """tkinter를 통해 실제 모니터 해상도 반환 (wx, hy)"""
    root = tk.Tk()
    root.withdraw()                     # 창을 숨긴 채로 정보만 가져옴
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return sw, sh


def load_image(path: str, label: str) -> np.ndarray:
    """이미지 로드 + None 체크"""
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] {label} 이미지를 불러올 수 없습니다: {path}")
        sys.exit(1)
    return img


def calc_display_scale(img_w: int, img_h: int,
                        sw: int, sh: int,
                        ratio: float = SCREEN_RATIO) -> float:
    """이미지가 화면 ratio 이내에 들어오도록 스케일 계산 (최대 1.0)"""
    return min((sw * ratio) / img_w,
               (sh * ratio) / img_h,
               1.0)


def show_centered_window(win_name: str, image: np.ndarray,
                          sw: int, sh: int) -> None:
    """창을 만들고 화면 중앙에 배치"""
    h, w = image.shape[:2]
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow(win_name, w, h)
    cv2.moveWindow(win_name, (sw - w) // 2, (sh - h) // 2)


def subpixel_peak(response: np.ndarray,
                   peak: tuple[int, int]) -> tuple[float, float]:
    """
    2차 포물선 보간(파라볼릭 피팅)으로 서브픽셀 정밀도 위치 계산.
    peak = (col, row)  →  반환값도 (col, row) 형식.
    """
    px, py = peak          # (col, row)
    rows, cols = response.shape

    # 경계면 처리 — 보간 불가 시 정수 위치 그대로 반환
    if px < 1 or px >= cols - 1 or py < 1 or py >= rows - 1:
        return float(px), float(py)

    # X 방향 (col)
    dx_num = response[py, px - 1] - response[py, px + 1]
    dx_den = 2.0 * (response[py, px - 1] - 2 * response[py, px] + response[py, px + 1])
    sub_x  = px + (dx_num / dx_den if abs(dx_den) > 1e-10 else 0.0)

    # Y 방향 (row)
    dy_num = response[py - 1, px] - response[py + 1, px]
    dy_den = 2.0 * (response[py - 1, px] - 2 * response[py, px] + response[py + 1, px])
    sub_y  = py + (dy_num / dy_den if abs(dy_den) > 1e-10 else 0.0)

    return sub_x, sub_y


def draw_result_overlay(img1: np.ndarray, img2: np.ndarray,
                         orig_pt: tuple[int, int],
                         found_pt: tuple[float, float],
                         dx: float, dy: float,
                         display_scale: float,
                         sw: int, sh: int) -> None:
    """
    img1(기준)과 img2(변형) 나란히 비교 표시.
    왼쪽: 기준 이미지에 원본 템플릿 위치 표시
    오른쪽: 변형 이미지에 매칭 위치 표시
    두 창 모두 화면 크기에 맞게 리사이즈.
    """
    # ── 기준 이미지 오버레이 ──────────────────────────────────
    vis1 = img1.copy()
    half = TEMPLATE_SIZE // 2
    ox, oy = orig_pt
    cv2.rectangle(vis1,
                  (ox - half, oy - half),
                  (ox + half, oy + half),
                  (0, 255, 0), 2)
    cv2.drawMarker(vis1, (ox, oy),
                   (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
    cv2.putText(vis1, "Reference", (ox - half, oy - half - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # ── 변형 이미지 오버레이 ─────────────────────────────────
    vis2 = img2.copy()
    fx, fy = int(round(found_pt[0])), int(round(found_pt[1]))
    cv2.rectangle(vis2,
                  (fx - half, fy - half),
                  (fx + half, fy + half),
                  (0, 0, 255), 2)
    cv2.drawMarker(vis2, (fx, fy),
                   (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    label = f"dx={dx:+.3f}px  dy={dy:+.3f}px"
    cv2.putText(vis2, label, (fx - half, fy - half - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # ── 리사이즈 후 표시 ─────────────────────────────────────
    s1 = calc_display_scale(*vis1.shape[:2][::-1], sw, sh)
    s2 = calc_display_scale(*vis2.shape[:2][::-1], sw, sh)

    r1 = cv2.resize(vis1,
                    (int(vis1.shape[1] * s1), int(vis1.shape[0] * s1)),
                    interpolation=cv2.INTER_LANCZOS4)
    r2 = cv2.resize(vis2,
                    (int(vis2.shape[1] * s2), int(vis2.shape[0] * s2)),
                    interpolation=cv2.INTER_LANCZOS4)

    show_centered_window("① Reference Image", r1, sw, sh)
    cv2.imshow("① Reference Image", r1)

    show_centered_window("② Deformed Image (Result)", r2, sw, sh)
    cv2.imshow("② Deformed Image (Result)", r2)


# ──────────────────────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────────────────────
def main() -> None:

    # 1. 이미지 로드
    img1 = load_image(IMG1_PATH, "기준(Reference)")
    img2 = load_image(IMG2_PATH, "변형(Deformed)")

    img_h, img_w = img1.shape[:2]
    sw, sh = get_screen_size()

    # 2. 표시용 스케일 계산 & 리사이즈
    scale      = calc_display_scale(img_w, img_h, sw, sh)
    disp_w     = int(img_w * scale)
    disp_h     = int(img_h * scale)
    img_display = cv2.resize(img1, (disp_w, disp_h),
                              interpolation=cv2.INTER_LANCZOS4)

    print(f"\n[INFO] 이미지 원본: {img_w}×{img_h}px")
    print(f"[INFO] 표시 크기:   {disp_w}×{disp_h}px  (비율 {scale*100:.1f}%)")

    # 3. ROI 선택 창
    WIN_SEL = "① Select Template Area — drag then press SPACE or ENTER"
    show_centered_window(WIN_SEL, img_display, sw, sh)
    print("\n[ACTION] 분석할 영역을 드래그로 선택한 뒤 ENTER / SPACE 를 누르세요.")
    print("         ESC 를 누르면 취소합니다.\n")

    roi = cv2.selectROI(WIN_SEL, img_display,
                         fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(WIN_SEL)

    # ROI 취소 or 빈 선택
    if roi[2] == 0 or roi[3] == 0:
        print("[WARN] 영역이 선택되지 않았습니다. 프로그램을 종료합니다.")
        sys.exit(0)

    # 4. 화면 좌표 → 원본 고해상도 좌표 변환
    rx, ry, rw, rh = [int(v / scale) for v in roi]

    # 선택 영역의 중심점 (템플릿 중심)
    cx = rx + rw // 2
    cy = ry + rh // 2

    # 5. 템플릿 추출 (고해상도)
    half     = TEMPLATE_SIZE // 2
    tx1      = max(0, cx - half)
    ty1      = max(0, cy - half)
    tx2      = min(img_w, tx1 + TEMPLATE_SIZE)
    ty2      = min(img_h, ty1 + TEMPLATE_SIZE)

    # 경계 초과 시 시작점 재조정 (크기 일정 유지)
    if tx2 - tx1 < TEMPLATE_SIZE:
        tx1 = max(0, tx2 - TEMPLATE_SIZE)
    if ty2 - ty1 < TEMPLATE_SIZE:
        ty1 = max(0, ty2 - TEMPLATE_SIZE)

    gray1   = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2   = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    template = gray1[ty1:ty2, tx1:tx2]

    actual_th, actual_tw = template.shape
    print(f"[INFO] 템플릿 크기: {actual_tw}×{actual_th}px  "
          f"(중심: {cx},{cy})")

    # 6. 탐색 영역 제한 (전체 이미지보다 탐색 범위를 좁혀 속도·정확도 향상)
    #    탐색 창: 템플릿 위치 ± SEARCH_MARGIN
    sx1 = max(0,       tx1 - SEARCH_MARGIN)
    sy1 = max(0,       ty1 - SEARCH_MARGIN)
    sx2 = min(img_w,   tx2 + SEARCH_MARGIN)
    sy2 = min(img_h,   ty2 + SEARCH_MARGIN)
    search_region = gray2[sy1:sy2, sx1:sx2]

    print(f"[INFO] 탐색 영역:  ({sx1},{sy1}) ~ ({sx2},{sy2})")

    # 7. 템플릿 매칭 (TM_CCOEFF_NORMED — 조명 변화에 강함)
    response = cv2.matchTemplate(search_region, template,
                                  cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(response)

    print(f"[INFO] 매칭 신뢰도: {max_val:.4f}  "
          f"(0.5 미만이면 결과를 신뢰하기 어렵습니다)")
    if max_val < 0.5:
        print("[WARN] 신뢰도가 낮습니다. 다른 영역을 선택하거나 "
              "이미지를 확인하세요.")

    # 8. 서브픽셀 정밀도
    if SUBPIXEL:
        sub_x, sub_y = subpixel_peak(response, max_loc)
    else:
        sub_x, sub_y = float(max_loc[0]), float(max_loc[1])

    # 탐색 영역 오프셋 적용 → img2 전체 좌표계로 변환
    found_x_in_img2 = sub_x + sx1          # img2에서 템플릿 좌상단 X
    found_y_in_img2 = sub_y + sy1          # img2에서 템플릿 좌상단 Y

    # 매칭된 위치의 중심 (img2 좌표계)
    found_cx = found_x_in_img2 + actual_tw / 2.0
    found_cy = found_y_in_img2 + actual_th / 2.0

    # 9. 변위 계산
    #    기준: 원본 img1에서 템플릿 중심  →  img2에서 매칭 중심
    dx = found_cx - (tx1 + actual_tw / 2.0)
    dy = found_cy - (ty1 + actual_th / 2.0)
    dist_px = np.hypot(dx, dy)

    # 10. 결과 출력
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  DIC 분석 결과  (원본 고해상도 기준, {'서브픽셀' if SUBPIXEL else '픽셀'} 정밀도)")
    print(sep)
    print(f"  템플릿 중심 (img1) : ({tx1 + actual_tw/2.0:.1f}, "
          f"{ty1 + actual_th/2.0:.1f}) px")
    print(f"  매칭 중심  (img2) : ({found_cx:.3f}, {found_cy:.3f}) px")
    print(f"  X축 변위           : {dx:+.4f} px")
    print(f"  Y축 변위           : {dy:+.4f} px")
    print(f"  직선 변위 (합성)   : {dist_px:.4f} px")
    print(f"  매칭 신뢰도        : {max_val:.4f}")
    print(sep)

    # 11. 결과 시각화 (기준·변형 이미지 나란히)
    draw_result_overlay(
        img1, img2,
        orig_pt    = (int(tx1 + actual_tw // 2), int(ty1 + actual_th // 2)),
        found_pt   = (found_cx, found_cy),
        dx=dx, dy=dy,
        display_scale=scale,
        sw=sw, sh=sh
    )

    print("\n[INFO] 아무 키나 누르면 종료합니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()