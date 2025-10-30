import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque

class HandTracker:
    """
    손가락을 감지해서 현재 모드와 좌표를 반환 (신철민 님 파트)
    """
    def __init__(self, history_len=5, draw_thresh=30, erase_thresh=100):
        # MediaPipe Hands 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.point_history = deque(maxlen=history_len)
        self.draw_threshold = draw_thresh
        self.erase_threshold = erase_thresh
        self.frame_width = 0
        self.frame_height = 0

    def get_gesture(self, frame):
        """
        프레임을 입력받아 모드, 좌표, 디버그 프레임을 반환
        """
        
        # frame 처리
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        # 디버깅용 frame 복사
        debug_frame = frame.copy() 

        # frame 크기 저장 (최초 1회)
        if self.frame_width == 0:
            self.frame_height, self.frame_width, _ = frame.shape

        # 손 감지
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 랜드마크 추출(엄지 끝, 검지 끝)
                lm_4 = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                lm_8 = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # 좌표 생성 (원본 프레임 기준)
                lm_4_x = int(lm_4.x * self.frame_width)
                lm_4_y = int(lm_4.y * self.frame_height)
                lm_8_x = int(lm_8.x * self.frame_width)
                lm_8_y = int(lm_8.y * self.frame_height)

                # 거리 계산
                distance = math.hypot(lm_8_x - lm_4_x, lm_8_y - lm_4_y)
                
                # 좌표 스무딩 (검지 기준)
                self.point_history.append((lm_8_x, lm_8_y))
                smooth_x = int(np.mean([pt[0] for pt in self.point_history]))
                smooth_y = int(np.mean([pt[1] for pt in self.point_history]))
                current_pt = (smooth_x, smooth_y)

                # 디버깅용 시각화
                cv2.circle(debug_frame, (smooth_x, smooth_y), 12, (255, 100, 0), 2) # 커서
                cv2.putText(debug_frame, f'{distance:.0f}', (lm_8_x, lm_8_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 모드 반환
                if distance >= self.erase_threshold:
                    return 'erase', current_pt, debug_frame
                elif distance <= self.draw_threshold:
                    return 'draw', current_pt, debug_frame
                else:
                    return 'move', current_pt, debug_frame # 'move'는 커서 이동
        
        # 손이 감지되지 않음
        self.point_history.clear()
        return 'none', (-1, -1), debug_frame

    def close(self):
        self.hands.close()
        print("HandTracker 리소스 해제됨.")