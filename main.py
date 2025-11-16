import cv2
import numpy as np
from module.handTracker import HandTracker
from module.UserMaskManager import UserMaskManager
from module.BackgroundManager import BackgroundManager

from module.utils import file_select_dialog
from module.shape_Recog import ShapeRecognizer

from module.keyboard_input import KeyboardInputManager
from module.overlay_hud import draw_hud

from module.view_manager import ViewManager

class VirtualBlackboard:
    """
    Manages and composites all layers (background, drawing, user).
    [MODIFIED] Based on the "black canvas(0) + color ink(1-255)" model.
    """

    def __init__(self, cap_w, cap_h, background_path=None):
        # Layer resolution (original)
        self.width = cap_w
        self.height = cap_h
        self.background_path = background_path

        # Low resolution for AI processing
        self.PROC_WIDTH = 640
        self.PROC_HEIGHT = int(self.PROC_WIDTH * (self.height / self.width))

        # [MODIFIED] Canvas with black background (normal)
        self.canvas = np.zeros(
            (self.height, self.width, 3), dtype=np.uint8
        )  # Black canvas

        # Class initialization
        self.hand_tracker = HandTracker(draw_thresh=30, erase_thresh=120)
        self.bg_module = UserMaskManager()  # ★ Load cvzone module
        self.bg_manager = BackgroundManager(self.width, self.height)

        # Add shape recognition module
        self.shape_recognizer = ShapeRecognizer(
            history_len=500,
            min_contour_area=500,
        )

        # [Layer 1] Background layer (black)
        self.background = self.bg_module.create_layer1_background(
            (self.height, self.width, 3), color=(0, 0, 0)  # Black blackboard
        )

        self.prev_draw_pt = (-1, -1)  # Previous coordinate for drawing lines
        self.drawing_mode = "normal"  # 'normal' or 'shape'

        # [MODIFIED] Drawing/erasing settings (based on black canvas)
        self.draw_color = (255, 255, 255)  # Ink: white (default)
        self.draw_thickness = 8
        self.erase_color = (0, 0, 0)  # Eraser: black (canvas background color)
        self.erase_thickness = 100

        # Per-page canvas management
        self.page_canvases = {}
        self.current_page_index = 0
        self.page_canvases[self.current_page_index] = self.canvas

        # For saving layers for PIP
        self.last_combined_bg = None
        self.last_frame = None

    def add_back_ground(self, source=None, color=(0, 0, 0)):
        self.background_path = source
        self.bg_manager.add_background(source=source, color=color)

        # Reset per-page canvases when a new background is loaded
        self.page_canvases = {}
        self.current_page_index = 0
        
        # [MODIFIED] Create the default canvas as 'black' (np.ones -> np.zeros)
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.page_canvases[self.current_page_index] = self.canvas

    def update(self, frame, drawing_enabled, user_mask_enabled):
        """
        Main update function called for every frame.
        """
        self._sync_canvas_with_page()

        # [MODIFIED] Prevent 't' key error: set default values for gesture_mode, point
        gesture_mode = 'none'
        point = (-1, -1)
        # debug_frame = frame # (if needed)

        if drawing_enabled:
            gesture_mode, point, debug_frame = self.hand_tracker.get_gesture(frame)
            self.update_canvas(gesture_mode, point)
        else:
            self.update_canvas('move', (-1, -1))

        # Create user mask
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

        # Final rendering
        output_frame = self.render(frame, self.canvas, user_mask)
        
        return output_frame, gesture_mode, point

    def update_canvas(self, mode, point):
        """
        Updates the drawing canvas (self.canvas) based on hand input.
        (This function already works correctly with the "black canvas" model)
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
                self.erase_color,  # Paints with 0 (black)
                self.erase_thickness,
            )
            self.prev_draw_pt = point

        else:  # 'move' or 'none'
            self.prev_draw_pt = (-1, -1)

    def render(self, frame, canvas, user_mask):
            """
            3-Layer composition logic (Modified: Optimized mask creation using cv2.inRange)
            Order: (1.Background -> 2.User) -> 3.Drawing
            """
            user_mask = np.ascontiguousarray(user_mask, dtype=np.uint8)

            # (Layer 1) Get background
            bg_view = (
                self.bg_manager.get_view()
                if self.background_path is not None
                else self.background
            )

            # (Layer 2) Cut out user
            user_part = cv2.bitwise_and(frame, frame, mask=user_mask)
            bg_mask = cv2.bitwise_not(user_mask)
            final_bg_part = cv2.bitwise_and(bg_view, bg_view, mask=bg_mask)

            # (Layer 1 + Layer 2) Composite
            bg_with_user = cv2.add(final_bg_part, user_part)

            # === [CORE OPTIMIZATION] ===
            # (Layer 3) Composite drawing canvas (based on black canvas)
            
            # 1. Mask out only the background part (0,0,0) of the canvas. (cvtColor+threshold -> inRange)
            #    This is the 'background mask' (mask_bg).
            mask_bg = cv2.inRange(canvas, (0, 0, 0), (0, 0, 0))
            
            # 2. Ink mask (ink=255, background=0)
            mask_ink = cv2.bitwise_not(mask_bg)
            # =======================

            # 3. Black out the area where the ink will be drawn on the (background+user) image.
            bg_part = cv2.bitwise_and(bg_with_user, bg_with_user, mask=mask_bg)
            
            # 4. Cut out only the ink part from the canvas.
            ink_part = cv2.bitwise_and(canvas, canvas, mask=mask_ink)

            # 5. (Punched background) + (ink) = Final composite
            output = cv2.add(bg_part, ink_part)

            # --- [Save PIP Layer] (Logic modified similarly) ---
            bg_part_pip = cv2.bitwise_and(bg_view, bg_view, mask=mask_bg)
            ink_part_pip = cv2.bitwise_and(canvas, canvas, mask=mask_ink)
            self.last_combined_bg = cv2.add(bg_part_pip, ink_part_pip)
            
            self.last_frame = frame.copy()

            return output

    def _sync_canvas_with_page(self):
        # ... (Page index logic remains the same) ...
        page_idx = 0
        if self.background_path is not None and self.bg_manager.mode == "pdf":
            page_idx = self.bg_manager.page_index
        
        if page_idx != self.current_page_index:
            # Save the canvas of the previous page
            self.page_canvases[self.current_page_index] = self.canvas

            # Load or create the canvas for the new page
            if page_idx in self.page_canvases:
                self.canvas = self.page_canvases[page_idx]
            else:
                # [MODIFIED] Create a new canvas as 'black' (np.ones -> np.zeros)
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.page_canvases[page_idx] = self.canvas

            self.current_page_index = page_idx

    def clear_canvas(self):
        """[MODIFIED] Initialize canvas to black"""
        self.canvas.fill(0) # Fill with 0 instead of 255
        print("Canvas cleared.")

    def update_shape_recognizer_color(self, color):
        """Pass the current pen color to the shape recognizer module"""
        self.shape_recognizer.set_draw_color(color)

    def close(self):
        """Release all resources"""
        self.hand_tracker.close()
        self.bg_module.close()


# Main function
def main():
    # Connect to webcam (high resolution)
    CAP_WIDTH, CAP_HEIGHT = 1280, 720
    bg_file_path = None
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(3, CAP_WIDTH)
    cap.set(4, CAP_HEIGHT)

    # Create main blackboard object
    blackboard = VirtualBlackboard(CAP_WIDTH, CAP_HEIGHT)

    blackboard.add_back_ground(bg_file_path)

    # Keyboard manager
    kb = KeyboardInputManager()

    # View mode manager
    view = ViewManager()

    window_name = "Virtual Blackboard (cvzone DNN)"
    cv2.namedWindow(window_name) 
    # start full screen
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, blackboard.bg_manager.on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame. (Stream end?)")
            break

        # Flip horizontally
        frame = cv2.flip(frame, 1)

        # Force frame to match blackboard resolution (to prevent size mismatch)
        frame = cv2.resize(frame, (blackboard.width, blackboard.height))

        # Get current toggle states from the keyboard manager (kb)
        draw_flag = kb.drawing_enabled
        mask_flag = kb.user_mask_enabled

        # Call the main update function (Virtual Blackboard 3-Layer composite)
        # Pass the read flags to the update function
        output_image, gesture_mode, point = blackboard.update(frame, draw_flag, mask_flag)
        # HUD + View mode
        display_image = view.compose(output_image, blackboard, kb)

        if draw_flag:
            # Finger pointer (debugging)
            debug_color = (255, 0, 255) # Draw (Magenta)
            if(gesture_mode == "erase"):
                debug_color = (255, 255, 0) # Erase (Cyan)
            elif(gesture_mode == "move"):
                debug_color = (0, 255, 255) # Move (Yellow)
            cv2.circle(display_image, point, 12, debug_color, cv2.FILLED)

        # Display output
        cv2.imshow(window_name, display_image)

        # Keyboard events (special key code constants)
        LEFT, UP, RIGHT, DOWN = 2424832, 2490368, 2555904, 2621440
        key = cv2.waitKeyEx(1)

        # Pass to keyboard manager first (additional features: color/thickness/record/snapshot/help)
        #  - Snapshot (P) saves the final screen including the HUD
        kb.handle_key(key, blackboard, current_frame_for_snapshot=display_image)

        # Apply manager state to the blackboard (pen color/thickness, etc.)
        kb.apply_to_blackboard(blackboard)

        # Update pen color in shape recognizer
        blackboard.update_shape_recognizer_color(blackboard.draw_color)

        # ===== Keep existing shortcut logic =====
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

        # ↑/↓ Zoom control (background zoom)
        elif key in (82, UP):  # ↑ Zoom In
            blackboard.bg_manager.zoom = min(blackboard.bg_manager.zoom * 1.1, 5.0)
        elif key in (84, DOWN):  # ↓ Zoom Out
            blackboard.bg_manager.zoom = max(blackboard.bg_manager.zoom * 0.9, 0.3)

        # 's' key to toggle shape mode (keep existing)
        elif key == ord("s"):
            if blackboard.drawing_mode == "normal":
                blackboard.drawing_mode = "shape"
                print("Shape Mode ON")
            else:
                blackboard.drawing_mode = "normal"
                print("Normal Mode ON")

            # Clear buffer and previous point to prevent correction of the last stroke upon mode switch
            blackboard.shape_recognizer.current_drawing_pts.clear()
            blackboard.prev_draw_pt = (-1, -1)
            blackboard.shape_recognizer.prev_mode = "none"

        # Toggle view mode (z key)
        elif key == ord("z"):
            view.toggle_mode()

        elif key == ord("x"):  # 'x' to turn off PDF
            blackboard.add_back_ground(None, color=(0, 0, 0))
            print("[BG] Reverted to solid color blackboard mode")

        # If recording, record the current frame (save after render -> HUD)
        kb.after_render(display_image)

    # Release resources
    blackboard.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    # TEST
