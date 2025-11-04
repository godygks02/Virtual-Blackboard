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
        입력 프레임(BGR)에서 사람(전경) 마스크를 생성하여
        8-bit 단일채널(0/255) C-연속 메모리로 반환.
        """
        # MediaPipe는 RGB 입력 기대
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmentor.selfieSegmentation.process(img_rgb)

        # 세그멘테이션 결과 없으면 바로 0 마스크 반환
        if results.segmentation_mask is None:
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        # float(0~1) → 이진(0/255)
        mask_float = results.segmentation_mask
        user_mask = (mask_float > threshold).astype(np.uint8) * 255

        # 단일채널/연속성/dtype 보장
        if user_mask.ndim == 3:
            user_mask = cv2.cvtColor(user_mask, cv2.COLOR_BGR2GRAY)
        user_mask = np.ascontiguousarray(user_mask, dtype=np.uint8)

        return user_mask


    def close(self):
        """리소스 해제"""
        if hasattr(self.segmentor, "selfieSegmentation"):
            self.segmentor.selfieSegmentation.close()
        print("BackgroundModule(cvzone) 리소스 해제됨.")
