import os
import cv2
import time
import datetime
import numpy as np

class KeyboardInputManager:
    """
    A manager that controls the state of the VirtualBlackboard (mode/color/thickness)
    and toggles for saving/recording/help via keyboard input.
    """
    def __init__(self):
        # Drawing state (reflected on the blackboard)
        self.pen_color = (255, 255, 255)  # Default white
        self.thickness = 8
        self.last_msg = ""                # HUD message (short feedback)
        
        # Help/HUD toggle
        self.help_on = False

        self.hud_on = True  # Toggle with '`' key (entire HUD On/Off)

        # === [NEW] Feature Toggle States ===
        self.drawing_enabled = True     # Toggle with 't' key (Hand recognition/drawing On/Off)
        self.user_mask_enabled = True   # Toggle with 'u' key (User mask On/Off)
        # ==============================

        # Recording state
        self.is_recording = False
        self.writer = None
        self.rec_fps = 30  # Adjust if necessary
        self.rec_path_dir = "recordings"
        self.cap_path_dir = "captures"
        os.makedirs(self.rec_path_dir, exist_ok=True)
        os.makedirs(self.cap_path_dir, exist_ok=True)

    # ---- Utils ----
    def _ts(self) -> str:
        # Filename-safe timestamp
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _start_recording(self, frame_w, frame_h):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(self.rec_path_dir, f"VB_{self._ts()}.mp4")
        self.writer = cv2.VideoWriter(out_path, fourcc, self.rec_fps, (frame_w, frame_h))
        self.is_recording = True
        self.last_msg = f"[REC ON] {out_path}"

    def _stop_recording(self):
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.is_recording = False
        self.last_msg = "[REC OFF] saved."

    def _save_snapshot(self, frame_bgr):
        out_path = os.path.join(self.cap_path_dir, f"CAP_{self._ts()}.png")
        cv2.imwrite(out_path, frame_bgr)
        self.last_msg = f"[SNAP] saved to {out_path}"

    # ---- Called from outside ----
    def after_render(self, output_image):
        """
        Receives the final rendered frame and records it if recording is active.
        """
        if self.is_recording and self.writer is not None:
            frame = output_image
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            self.writer.write(frame)

    def apply_to_blackboard(self, blackboard):
        """
        Applies the current manager state (pen_color, thickness) to the blackboard.
        """
        blackboard.draw_color = self.pen_color
        blackboard.draw_thickness = self.thickness

    def handle_key(self, key, blackboard, current_frame_for_snapshot=None):
        """
        Handles key input, changes internal state, and performs side effects (recording/snapshot) if necessary.
        """
        if key == -1:
            return

        # ====== Additional Features ======
        # Pen color shortcuts
        if key == ord('w'):  # white
            self.pen_color = (255, 255, 255); self.last_msg = "Pen: WHITE"
        elif key == ord('r'):  # red
            self.pen_color = (0, 0, 255); self.last_msg = "Pen: RED"
        elif key == ord('g'):  # green
            self.pen_color = (0, 255, 0); self.last_msg = "Pen: GREEN"
        elif key == ord('b'):  # blue
            self.pen_color = (255, 0, 0); self.last_msg = "Pen: BLUE"
        elif key == ord('y'):  # yellow
            self.pen_color = (0, 255, 255); self.last_msg = "Pen: YELLOW"

        # Adjust pen thickness
        elif key in (ord('+'), ord('=')):  # Consider '=' for keyboard layout compatibility
            self.thickness = min(self.thickness + 2, 60)
            self.last_msg = f"Thickness: {self.thickness}"
        elif key in (ord('-'), ord('_')):
            self.thickness = max(self.thickness - 2, 1)
            self.last_msg = f"Thickness: {self.thickness}"

        # Presets 1-5
        elif key == ord('1'):
            self.pen_color, self.thickness = (255,255,255), 6;  self.last_msg = "Preset1: chalk"
        elif key == ord('2'):
            self.pen_color, self.thickness = (0,0,255), 10;    self.last_msg = "Preset2: red marker"
        elif key == ord('3'):
            self.pen_color, self.thickness = (0,255,0), 12;    self.last_msg = "Preset3: green marker"
        elif key == ord('4'):
            self.pen_color, self.thickness = (255,0,0), 12;    self.last_msg = "Preset4: blue marker"
        elif key == ord('5'):
            self.pen_color, self.thickness = (0,255,255), 18;  self.last_msg = "Preset5: highlighter"

        # Toggle recording: V
        elif key == ord('v'):
            if not self.is_recording:
                self._start_recording(blackboard.width, blackboard.height)
            else:
                self._stop_recording()

        # Snapshot: P
        elif key == ord('p'):
            if current_frame_for_snapshot is not None:
                self._save_snapshot(current_frame_for_snapshot)

        # Help: H
        elif key == ord('h'):
            self.help_on = not self.help_on
            self.last_msg = "Help ON" if self.help_on else "Help OFF"

        # === [NEW] Feature Toggle Keys ===
        # 't' : Toggle hand recognition (drawing/erasing)
        elif key == ord('t'):
            self.drawing_enabled = not self.drawing_enabled
            if self.drawing_enabled:
                self.last_msg = "Hand Tracking ON"
            else:
                self.last_msg = "Hand Tracking OFF"

        # 'u' : Toggle user mask (background removal)
        elif key == ord('u'):
            self.user_mask_enabled = not self.user_mask_enabled
            if self.user_mask_enabled:
                self.last_msg = "User Mask ON"
            else:
                self.last_msg = "User Mask OFF"
        
        # '`' (backtick, above Tab) : Toggle entire HUD
        elif key == ord('`'):
            self.hud_on = not self.hud_on
            self.last_msg = "HUD ON" if self.hud_on else "HUD OFF"
        # ==============================

    # Simple help string
    def help_lines(self):
        return [
            "Shortcuts (KeyboardInputManager):",
            "  w/r/g/b/y : pen color (white/red/green/blue/yellow)",
            "  +/-       : pen thickness up/down",
            "  1..5      : presets",
            "  v         : start/stop recording (MP4)",
            "  p         : snapshot (PNG)",
            "  h         : toggle help",
            "  t         : toggle hand tracking (Draw ON/OFF)", 
            "  u         : toggle user mask (Show/Hide User)", 
            "  `         : toggle HUD (Show/Hide this info)",
            "",
            "Existing keys (main.py):",
            "  s         : toggle shape mode",
            "  c         : clear canvas",
            "  f         : open background file",
            "  x         : close background file",
            "  z         : toggle view mode (PIP)",
            "  arrow L/R : prev/next page(or use a/d)",
            "  arrow U/D : zoom in/out",
            "  q         : quit",
        ]
