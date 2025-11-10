import cv2
import numpy as np
import mediapipe as mp
from cvzone.SelfiSegmentationModule import SelfiSegmentation


class BackgroundModule:
    """
    (Layer 1) Background Creation + (Layer 3) User Segmentation with cvzone(MediaPipe)
    Controls MediaPipe more easily using the cvzone library.
    """

    def __init__(self, model=1):
        # Initialize cvzone Segmenter
        self.segmentor = SelfiSegmentation(model=model)
        print("cvzone(MediaPipe) DNN model loaded successfully.")

    def create_layer1_background(self, frame_shape, color=(0, 0, 0)):
        """(Layer 1) Create virtual blackboard background"""
        return np.full(frame_shape, color, dtype=np.uint8)

    def create_layer3_mask(self, frame, threshold=0.62):
        """
        Creates a person (foreground) mask from the input frame (BGR) and
        returns it as an 8-bit single-channel (0/255) C-contiguous memory.
        """
        # MediaPipe expects RGB input
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmentor.selfieSegmentation.process(img_rgb)

        # If no segmentation result, return a zero mask immediately
        if results.segmentation_mask is None:
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        # float(0~1) -> binary(0/255)
        mask_float = results.segmentation_mask
        user_mask = (mask_float > threshold).astype(np.uint8) * 255

        # Ensure single-channel/continuity/dtype
        if user_mask.ndim == 3:
            user_mask = cv2.cvtColor(user_mask, cv2.COLOR_BGR2GRAY)
        user_mask = np.ascontiguousarray(user_mask, dtype=np.uint8)

        return user_mask


    def close(self):
        """Release resources"""
        if hasattr(self.segmentor, "selfieSegmentation"):
            self.segmentor.selfieSegmentation.close()
        print("BackgroundModule(cvzone) resources released.")
