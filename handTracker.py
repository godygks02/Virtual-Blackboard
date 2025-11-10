import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque


class HandTracker:
    """
    Detects fingers and returns the current mode and coordinates.
    """

    def __init__(self, history_len=5, draw_thresh=30, erase_thresh=150):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.point_history = deque(maxlen=history_len)
        self.draw_threshold = draw_thresh
        self.erase_threshold = erase_thresh
        self.frame_width = 0
        self.frame_height = 0

    def get_gesture(self, frame):
        """
        Takes a frame as input and returns mode, coordinates, and a debug frame.
        """

        # Process frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        # Copy frame for debugging
        debug_frame = frame.copy()

        # Save frame size (once at the beginning)
        if self.frame_width == 0:
            self.frame_height, self.frame_width, _ = frame.shape

        # Hand detection
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks (thumb tip, index finger tip)
                lm_4 = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                lm_8 = hand_landmarks.landmark[
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                ]

                # Generate coordinates (based on original frame)
                lm_4_x = int(lm_4.x * self.frame_width)
                lm_4_y = int(lm_4.y * self.frame_height)
                lm_8_x = int(lm_8.x * self.frame_width)
                lm_8_y = int(lm_8.y * self.frame_height)

                # Calculate distance
                distance = math.hypot(lm_8_x - lm_4_x, lm_8_y - lm_4_y)

                # Coordinate smoothing (based on index finger)
                self.point_history.append((lm_8_x, lm_8_y))
                smooth_x = int(np.mean([pt[0] for pt in self.point_history]))
                smooth_y = int(np.mean([pt[1] for pt in self.point_history]))
                current_pt = (smooth_x, smooth_y)

                # Visualization for debugging
                cv2.circle(
                    debug_frame, (smooth_x, smooth_y), 12, (255, 100, 0), 2
                )  # Cursor
                cv2.putText(
                    debug_frame,
                    f"{distance:.0f}",
                    (lm_8_x, lm_8_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Return mode
                if distance >= self.erase_threshold:
                    return "erase", current_pt, debug_frame
                elif distance <= self.draw_threshold:
                    return "draw", current_pt, debug_frame
                else:
                    return "move", current_pt, debug_frame  # 'move' is for cursor movement

        # Hand not detected
        self.point_history.clear()
        return "none", (-1, -1), debug_frame

    def close(self):
        self.hands.close()
        print("HandTracker resources released.")
