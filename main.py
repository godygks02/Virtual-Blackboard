import cv2
import numpy as np
from handTracker import HandTracker 
from BackGroundModule import BackgroundModule 

class VirtualBlackboard:
    """
    모든 레이어(배경, 드로잉, 사용자) 관리 및 합성
    """
    def __init__(self, cap_w, cap_h):
        # 레이어 해상도 (원본)
        self.width = cap_w
        self.height = cap_h
        
        # AI 처리를 위한 저해상도
        self.PROC_WIDTH = 640
        self.PROC_HEIGHT = int(self.PROC_WIDTH * (self.height / self.width))
        
        # 검은색 배경의 캔버스
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8) # 검은색 캔버스
        
        # 클래스 초기화
        self.hand_tracker = HandTracker(draw_thresh=30, erase_thresh=80)
        self.bg_module = BackgroundModule() # ★ cvzone 모듈 로드
        
        # [Layer 1] 배경 레이어 (검은색)
        self.background = self.bg_module.create_layer1_background(
            (self.height, self.width, 3), 
            color=(0, 0, 0) # 검은색 칠판
        )
        
        self.prev_draw_pt = (-1, -1) # 선 그리기를 위한 이전 좌표
        
        # 그리기/지우기 설정
        self.draw_color = (255, 255, 255) # 흰색
        self.draw_thickness = 8
        self.erase_color = (0, 0, 0) # 캔버스 배경색 (검은색)
        self.erase_thickness = 100

    def update(self, frame):
        """
        매 프레임마다 호출되는 메인 업데이트 함수
        """
        
        # (손) - 원본 고해상도 프레임으로 처리
        gesture_mode, point, debug_frame = self.hand_tracker.get_gesture(frame)

        # (드로잉) - 원본 해상도 캔버스에 그림
        self.update_canvas(gesture_mode, point)

        # 사용자 마스크 생성
        # 성능을 위해 640p로 축소하여 AI 처리
        frame_small = cv2.resize(frame, (self.PROC_WIDTH, self.PROC_HEIGHT))
        
        # AI가 640p 마스크 생성
        user_mask_small = self.bg_module.create_layer3_mask(frame_small, threshold=0.62)
        
        # 마스크를 원본 해상도(1280x720)로 다시 확대
        user_mask = cv2.resize(user_mask_small, (self.width, self.height))

        # 최종 렌더링 (3-Layer 합성)
        output_frame = self.render(frame, self.canvas, user_mask)
        
        return output_frame

    def update_canvas(self, mode, point):
        """
        입력(손)에 따라 드로잉 캔버스(self.canvas)를 업데이트
        """
        if mode == 'draw':
            if self.prev_draw_pt == (-1, -1):
                self.prev_draw_pt = point
            cv2.line(self.canvas, self.prev_draw_pt, point, self.draw_color, self.draw_thickness)
            self.prev_draw_pt = point
            
        elif mode == 'erase':
            if self.prev_draw_pt == (-1, -1):
                self.prev_draw_pt = point
            cv2.line(self.canvas, self.prev_draw_pt, point, self.erase_color, self.erase_thickness)
            self.prev_draw_pt = point
        
        else: # 'move' 또는 'none'
            self.prev_draw_pt = (-1, -1) # 선 끊기

    def render(self, frame, canvas, user_mask):
        """
        3-Layer 합성 로직 (검은색 캔버스 기준)
        순서: (1.배경 + 2.드로잉) -> 3.사용자
        """
        
        # (Layer 1 + Layer 2)
        # 검은색 배경(Layer 1)과 검은색 캔버스(Layer 2)를 합침
        # (배경이 0, 캔버스도 0이므로 add 연산으로 그림만 합쳐짐)
        combined_bg = cv2.add(self.background, canvas)

        # (Layer 3)
        # 원본 프레임에서 사용자 부분만 오려내기
        user_part = cv2.bitwise_and(frame, frame, mask=user_mask)
        
        # (칠판+그림)에서 사용자 아닌 배경 부분만 오려내기
        bg_mask = cv2.bitwise_not(user_mask)
        final_bg_part = cv2.bitwise_and(combined_bg, combined_bg, mask=bg_mask)

        # (칠판+그림 배경) + (사람)
        output = cv2.add(final_bg_part, user_part)
        
        return output

    def clear_canvas(self):
        """캔버스 검은색으로 초기화"""
        self.canvas.fill(0)
        print("Canvas cleared.")

    def close(self):
        """모든 리소스 해제"""
        self.hand_tracker.close()
        self.bg_module.close()

# 메인 함수
def main():
    # 웹캠 연결 (고해상도)
    CAP_WIDTH, CAP_HEIGHT = 1280, 720
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("오류: 카메라를 열 수 없습니다.")
        return
        
    cap.set(3, CAP_WIDTH) 
    cap.set(4, CAP_HEIGHT)

    # 메인 블랙보드 객체 생성
    blackboard = VirtualBlackboard(CAP_WIDTH, CAP_HEIGHT)
    
    window_name = 'Virtual Blackboard (cvzone DNN)'
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽어올 수 없습니다. (스트림 종료?)")
            break
        
        # 좌우 반전
        frame = cv2.flip(frame, 1) 

        # 메인 업데이트 함수 호출
        output_image = blackboard.update(frame)

        cv2.imshow(window_name, output_image)
        
        # 키보드 이벤트
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            blackboard.clear_canvas()

    # 리소스 해제
    blackboard.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


    #TEST