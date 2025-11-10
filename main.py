import cv2
import numpy as np
from handTracker import HandTracker
from BackGroundModule import BackgroundModule
from BackGroundManager import BackgroundManager

from utils import file_select_dialog
from handTracker import HandTracker
from BackGroundModule import BackgroundModule
from shape_Recog import ShapeRecognizer

from keyboard_input import KeyboardInputManager
from overlay_hud import draw_hud

from view_manager import ViewManager

class VirtualBlackboard:
    """
    모든 레이어(배경, 드로잉, 사용자) 관리 및 합성
    [수정] "검은색 캔버스(0) + 컬러 잉크(1-255)" 모델 기준
    """

    def __init__(self, cap_w, cap_h, background_path=None):
        # 레이어 해상도 (원본)
        self.width = cap_w
        self.height = cap_h
        self.background_path = background_path

        # AI 처리를 위한 저해상도
        self.PROC_WIDTH = 640
        self.PROC_HEIGHT = int(self.PROC_WIDTH * (self.height / self.width))

        # [수정] 검은색 배경의 캔버스 (정상)
        self.canvas = np.zeros(
            (self.height, self.width, 3), dtype=np.uint8
        )  # 검은색 캔버스

        # 클래스 초기화
        self.hand_tracker = HandTracker(draw_thresh=30, erase_thresh=120)
        self.bg_module = BackgroundModule()  # ★ cvzone 모듈 로드
        self.bg_manager = BackgroundManager(self.width, self.height)

        # 도형 인식 모듈 추가
        self.shape_recognizer = ShapeRecognizer(
            history_len=500,
            min_contour_area=500,
        )

        # [Layer 1] 배경 레이어 (검은색)
        self.background = self.bg_module.create_layer1_background(
            (self.height, self.width, 3), color=(0, 0, 0)  # 검은색 칠판
        )

        self.prev_draw_pt = (-1, -1)  # 선 그리기를 위한 이전 좌표
        self.drawing_mode = "normal"  # 'normal' 또는 'shape'

        # [수정] 그리기/지우기 설정 (검은색 캔버스 기준)
        self.draw_color = (255, 255, 255)  # 잉크: 흰색 (기본값)
        self.draw_thickness = 8
        self.erase_color = (0, 0, 0)  # 지우개: 검은색 (캔버스 배경색)
        self.erase_thickness = 100

        # 페이지별 캔버스 관리
        self.page_canvases = {}
        self.current_page_index = 0
        self.page_canvases[self.current_page_index] = self.canvas

        # PIP용 레이어 저장용
        self.last_combined_bg = None
        self.last_frame = None

    def add_back_ground(self, source=None, color=(0, 0, 0)):
        self.background_path = source
        self.bg_manager.add_background(source=source, color=color)

        # 새 배경 로딩 시 페이지별 캔버스 리셋
        self.page_canvases = {}
        self.current_page_index = 0
        
        # [수정] 기본 캔버스를 '검은색'으로 생성 (np.ones -> np.zeros)
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.page_canvases[self.current_page_index] = self.canvas

    def update(self, frame, drawing_enabled, user_mask_enabled):
        """
        매 프레임마다 호출되는 메인 업데이트 함수
        """
        self._sync_canvas_with_page()

        # [수정] 't'키 오류 방지: gesture_mode, point 기본값 설정
        gesture_mode = 'none'
        point = (-1, -1)
        # debug_frame = frame # (필요 시)

        if drawing_enabled:
            gesture_mode, point, debug_frame = self.hand_tracker.get_gesture(frame)
            self.update_canvas(gesture_mode, point)
        else:
            self.update_canvas('move', (-1, -1))

        # 사용자 마스크 생성
        if user_mask_enabled:
            frame_small = cv2.resize(frame, (self.PROC_WIDTH, self.PROC_HEIGHT))
            user_mask_small = self.bg_module.create_layer3_mask(frame_small, threshold=0.62)
            user_mask = cv2.resize(
                user_mask_small, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )
            if user_mask.ndim == 3:
                user_mask = cv2.cvtColor(user_mask, cv2.COLOR_BGR2GRAY)
            user_mask = np.ascontiguousarray(user_mask, dtype=np.uint8)
        else:
            user_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # 최종 렌더링
        output_frame = self.render(frame, self.canvas, user_mask)
        
        # [수정] 't'키 오류 방지: 디버깅 코드를 'if drawing_enabled:' 블록 안으로 이동
        if drawing_enabled:
            # 손가락 포인터 (디버깅)
            debug_color = (255, 0, 255) # Draw (Magenta)
            if(gesture_mode == "erase"):
                debug_color = (255, 255, 0) # Erase (Cyan)
            elif(gesture_mode == "move"):
                debug_color = (0, 255, 255) # Move (Yellow)
            cv2.circle(output_frame, point, 12, debug_color, cv2.FILLED)

        return output_frame

    def update_canvas(self, mode, point):
        """
        입력(손)에 따라 드로잉 캔버스(self.canvas)를 업데이트
        (이 함수는 "검은색 캔버스" 모델에서 이미 올바르게 작동합니다)
        """
        is_shape_recognized = False

        if self.drawing_mode == "shape":
            if mode == "draw":
                self.shape_recognizer.add_point(point)
            if self.shape_recognizer.prev_mode == "draw" and mode in ("move", "none", "erase"):
                is_shape_recognized = self.shape_recognizer.process_drawing(
                    mode, self.canvas
                )
            self.shape_recognizer.prev_mode = mode

        if is_shape_recognized:
            self.prev_draw_pt = (-1, -1)
            return
        
        if mode == "draw":
            if self.prev_draw_pt == (-1, -1):
                self.prev_draw_pt = point
            cv2.line(
                self.canvas,
                self.prev_draw_pt,
                point,
                self.draw_color,
                self.draw_thickness,
            )
            self.prev_draw_pt = point

        elif mode == "erase" and self.drawing_mode != "shape":
            if self.prev_draw_pt == (-1, -1):
                self.prev_draw_pt = point
            cv2.line(
                self.canvas,
                self.prev_draw_pt,
                point,
                self.erase_color,  # 0 (검은색)으로 칠함
                self.erase_thickness,
            )
            self.prev_draw_pt = point

        else:  # 'move' 또는 'none'
            self.prev_draw_pt = (-1, -1)

    def render(self, frame, canvas, user_mask):
            """
            3-Layer 합성 로직 (수정: cv2.inRange를 사용해 마스크 생성 최적화)
            순서: (1.배경 -> 2.사용자) -> 3.드로잉
            """
            user_mask = np.ascontiguousarray(user_mask, dtype=np.uint8)

            # (Layer 1) 배경 가져오기
            bg_view = (
                self.bg_manager.get_view()
                if self.background_path is not None
                else self.background
            )

            # (Layer 2) 사용자 오려내기
            user_part = cv2.bitwise_and(frame, frame, mask=user_mask)
            bg_mask = cv2.bitwise_not(user_mask)
            final_bg_part = cv2.bitwise_and(bg_view, bg_view, mask=bg_mask)

            # (Layer 1 + Layer 2) 합성
            bg_with_user = cv2.add(final_bg_part, user_part)

            # === [핵심 최적화] ===
            # (Layer 3) 드로잉 캔버스 합성 (검은색 캔버스 기준)
            
            # 1. 캔버스의 배경(0,0,0) 부분만 마스크로 따냅니다. (cvtColor+threshold -> inRange)
            #    이것이 '배경 마스크' (mask_bg)입니다.
            mask_bg = cv2.inRange(canvas, (0, 0, 0), (0, 0, 0))
            
            # 2. 잉크 마스크 (잉크=255, 배경=0)
            mask_ink = cv2.bitwise_not(mask_bg)
            # =======================

            # 3. (배경+사람)에서 잉크가 그려질 부분을 0(검은색)으로 지웁니다.
            bg_part = cv2.bitwise_and(bg_with_user, bg_with_user, mask=mask_bg)
            
            # 4. 캔버스에서 잉크 부분만 오려냅니다.
            ink_part = cv2.bitwise_and(canvas, canvas, mask=mask_ink)

            # 5. (구멍 뚫린 배경) + (잉크) = 최종 합성
            output = cv2.add(bg_part, ink_part)

            # --- [PIP 레이어 저장] (로직 동일하게 수정) ---
            bg_part_pip = cv2.bitwise_and(bg_view, bg_view, mask=mask_bg)
            ink_part_pip = cv2.bitwise_and(canvas, canvas, mask=mask_ink)
            self.last_combined_bg = cv2.add(bg_part_pip, ink_part_pip)
            
            self.last_frame = frame.copy()

            return output

    def _sync_canvas_with_page(self):
        # ... (페이지 인덱스 로직 동일) ...
        page_idx = 0
        if self.background_path is not None and self.bg_manager.mode == "pdf":
            page_idx = self.bg_manager.page_index
        
        if page_idx != self.current_page_index:
            # 이전 페이지 캔버스 저장
            self.page_canvases[self.current_page_index] = self.canvas

            # 새 페이지 캔버스 불러오기 또는 생성
            if page_idx in self.page_canvases:
                self.canvas = self.page_canvases[page_idx]
            else:
                # [수정] 새 캔버스를 '검은색'으로 생성 (np.ones -> np.zeros)
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.page_canvases[page_idx] = self.canvas

            self.current_page_index = page_idx

    def clear_canvas(self):
        """[수정] 캔버스 검은색으로 초기화"""
        self.canvas.fill(0) # 255 대신 0으로 채우기
        print("Canvas cleared.")

    def update_shape_recognizer_color(self, color):
        """현재 펜 색상을 도형 인식 모듈에 전달"""
        self.shape_recognizer.set_draw_color(color)

    def close(self):
        """모든 리소스 해제"""
        self.hand_tracker.close()
        self.bg_module.close()


# 메인 함수
def main():
    # 웹캠 연결 (고해상도)
    CAP_WIDTH, CAP_HEIGHT = 1280, 720
    bg_file_path = None
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("오류: 카메라를 열 수 없습니다.")
        return

    cap.set(3, CAP_WIDTH)
    cap.set(4, CAP_HEIGHT)

    # 메인 블랙보드 객체 생성
    blackboard = VirtualBlackboard(CAP_WIDTH, CAP_HEIGHT)

    blackboard.add_back_ground(bg_file_path)

    # 키보드 매니저
    kb = KeyboardInputManager()

    # 보기 모드 매니저
    view = ViewManager()

    window_name = "Virtual Blackboard (cvzone DNN)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, blackboard.bg_manager.on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽어올 수 없습니다. (스트림 종료?)")
            break

        # 좌우 반전
        frame = cv2.flip(frame, 1)

        # 프레임을 블랙보드 해상도에 강제 맞춤 (크기 불일치 방지)
        frame = cv2.resize(frame, (blackboard.width, blackboard.height))

        # 키보드 매니저(kb)에서 현재 토글 상태 가져옴
        draw_flag = kb.drawing_enabled
        mask_flag = kb.user_mask_enabled

        # 메인 업데이트 함수 호출 (가상 칠판 3-Layer 합성)
        # 읽어온 플래그를 update 함수에 전달
        output_image = blackboard.update(frame, draw_flag, mask_flag)
        # HUD + 보기 모드
        display_image = view.compose(output_image, blackboard, kb)

        # 화면 출력
        cv2.imshow(window_name, display_image)

        # 키보드 이벤트 (특수키 코드 상수)
        LEFT, UP, RIGHT, DOWN = 2424832, 2490368, 2555904, 2621440
        key = cv2.waitKeyEx(1)

        # 키보드 매니저에 먼저 전달 (추가 기능: 색/굵기/녹화/스냅샷/도움말)
        #  - 스냅샷(P)은 HUD 포함된 최종 화면을 저장
        kb.handle_key(key, blackboard, current_frame_for_snapshot=display_image)

        # 매니저 상태를 블랙보드에 반영 (펜 색상/굵기 등)
        kb.apply_to_blackboard(blackboard)

        # 도형인식에 펜 색상 업데이트
        blackboard.update_shape_recognizer_color(blackboard.draw_color)

        # ===== 기존 단축키 로직 유지 =====
        if key == ord("q"):
            break
        elif key == ord("c"):
            blackboard.clear_canvas()
        elif key in (81, LEFT, ord("a")):
            blackboard.bg_manager.prev_page()
            print("[DEBUG] MOVE PREV PAGE")
        elif key in (83, RIGHT, ord("d")):
            blackboard.bg_manager.next_page()
            print("[DEBUG] MOVE NEXT PAGE")
        elif key == ord("f"):
            selected_file_path = file_select_dialog()
            if selected_file_path != "":
                bg_file_path = selected_file_path
                blackboard.add_back_ground(bg_file_path)

        # ↑/↓ 줌 조절 (배경 확대/축소)
        elif key in (82, UP):  # ↑ 줌 인
            blackboard.bg_manager.zoom = min(blackboard.bg_manager.zoom * 1.1, 5.0)
        elif key in (84, DOWN):  # ↓ 줌 아웃
            blackboard.bg_manager.zoom = max(blackboard.bg_manager.zoom * 0.9, 0.3)

        # 's' 키로 도형 모드 전환 (기존 유지)
        elif key == ord("s"):
            if blackboard.drawing_mode == "normal":
                blackboard.drawing_mode = "shape"
                print("도형 그리기 모드 활성화 (Shape Mode ON)")
            else:
                blackboard.drawing_mode = "normal"
                print("일반 그리기 모드 활성화 (Normal Mode ON)")

            # 모드 전환 시 이전 획이 보정되는 것을 방지하기 위해 버퍼와 이전 좌표 초기화
            blackboard.shape_recognizer.current_drawing_pts.clear()
            blackboard.prev_draw_pt = (-1, -1)
            blackboard.shape_recognizer.prev_mode = "none"

        # 보기 모드 토글 (z키)
        elif key == ord("z"):
            view.toggle_mode()

        elif key == ord("x"):  # 'x'로 PDF 끄기
            blackboard.add_back_ground(None, color=(0, 0, 0))
            print("[BG] 단색 칠판 모드로 복귀")

        # 녹화 중이면 현재 프레임 기록 (렌더 → HUD 이후 저장)
        kb.after_render(display_image)

    # 리소스 해제
    blackboard.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    # TEST
