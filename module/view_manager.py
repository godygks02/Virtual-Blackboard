# view_manager.py
import cv2
import numpy as np
from .overlay_hud import draw_hud


class ViewManager:
    """
    - Draw HUD
    - Pass BG error messages
    - View modes (normal / pip):
        normal: Full screen as before
        pip: User in a small window at the bottom right, main screen is 'blackboard without user'
    """

    def __init__(self):
        self.view_mode = "normal"   # "normal" or "pip"

    def toggle_mode(self):
        # Toggle view mode
        self.view_mode = "pip" if self.view_mode == "normal" else "normal"
        print(f"[VIEW] mode = {self.view_mode}")

    def compose(self, base_frame, blackboard, kb_manager):
        """
        - base_frame: Result of VirtualBlackboard.update(frame) (the previous final screen)
        - blackboard: Uses last_combined_bg / last_frame
        - Returns: The final display frame
        """

        # If there is a BG error message, get it and pass it to the HUD (None otherwise)
        extra_msg = getattr(blackboard.bg_manager, "last_error", None)

        # Default is HUD on top of the original final frame
        frame_for_hud = base_frame

        # In pip mode: blackboard without person + PIP camera
        if self.view_mode == "pip":
            # Prioritize blackboard without person (if not available, use the existing final frame)
            if blackboard.last_combined_bg is not None:
                base = blackboard.last_combined_bg.copy()
            else:
                base = base_frame.copy()

            h, w = base.shape[:2]

            if blackboard.last_frame is not None:
                cam = blackboard.last_frame.copy()

                # 1) Set PIP size (about 1/4 of the screen height)
                pip_h = int(h * 0.25)
                pip_w = int(cam.shape[1] * (pip_h / cam.shape[0]))
                pip = cv2.resize(cam, (pip_w, pip_h))

                # 2) Create a circular mask (centered circle)
                mask = np.zeros((pip_h, pip_w), dtype=np.uint8)
                radius = min(pip_w, pip_h) // 2
                center = (pip_w // 2, pip_h // 2)
                cv2.circle(mask, center, radius, 255, -1)

                # 3) Circularly cropped face PIP
                pip_fg = cv2.bitwise_and(pip, pip, mask=mask)

                # 4) Placement position: bottom right
                margin = 20
                y2 = h - margin
                y1 = y2 - pip_h
                x2 = w - margin
                x1 = x2 - pip_w

                # (If you want to place it in the center of the screen)
                # cx, cy = w // 2, h // 2
                # x1 = cx - pip_w // 2
                # x2 = cx + pip_w // 2
                # y1 = cy - pip_h // 2
                # y2 = cy + pip_h // 2

                # 5) Clear the circle area from the background ROI
                roi = base[y1:y2, x1:x2]
                mask_inv = cv2.bitwise_not(mask)  # Mask that leaves only the area outside the circle
                bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                # 6) Composite background + face
                dst = cv2.add(bg, pip_fg)
                base[y1:y2, x1:x2] = dst

            frame_for_hud = base
        else:
            frame_for_hud = base_frame

        final = draw_hud(frame_for_hud, blackboard, kb_manager, extra_msg=extra_msg)
        return final
