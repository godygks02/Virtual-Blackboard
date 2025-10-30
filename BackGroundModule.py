import cv2
import numpy as np
import mediapipe as mp
from cvzone.SelfiSegmentationModule import SelfiSegmentation

class BackgroundModule:
    """
    (Layer 1) 배경 생성 + (Layer 3) cvzone(MediaPipe) 사용자 분리
    cvzone 라이브러리를 사용해 MediaPipe를 더 쉽게 제어합니다.
    """
    def __init__(self, model=1):
        # cvzone Segmenter 초기화
        self.segmentor = SelfiSegmentation(model=model)
        print("cvzone(MediaPipe) DNN 모델 로드 완료.")

    def create_layer1_background(self, frame_shape, color=(0, 0, 0)):
        """(Layer 1) 가상 칠판 배경 생성"""
        return np.full(frame_shape, color, dtype=np.uint8)

    def create_layer3_mask(self, frame, threshold=0.62):
        """
        (Layer 3) DNN을 사용해 '사람' 마스크를 생성
        
        cvzone의 removeBG 함수 로직을 재구성하여 '마스크'만 반환합니다.
        """
        
        # AI 모델 실행 (MediaPipe의 원본 mask를 얻기 위함)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.segmentor.selfieSegmentation.process(img_rgb)

        
        if results.segmentation_mask is None:
            # 사람이 감지되지 않으면 빈 마스크 반환
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # 마스크 결과(float 0.0 ~ 1.0)를 흑백(0/255)으로 변환
        mask_float = results.segmentation_mask
        
        # 사용자가 제공한 오픈소스 코드의 임계값(0.62) 사용
        user_mask = (mask_float > threshold).astype(np.uint8) * 255
        
        return user_mask
    
    def close(self):
        """리소스 해제"""
        if hasattr(self.segmentor, 'selfieSegmentation'):
             self.segmentor.selfieSegmentation.close()
        print("BackgroundModule(cvzone) 리소스 해제됨.")