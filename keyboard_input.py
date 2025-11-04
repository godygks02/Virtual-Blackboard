# keyboard_input.py
import os
import cv2
import time
import datetime
import numpy as np

class KeyboardInputManager:
    """
    키보드 입력을 통해 VirtualBlackboard의 상태(모드/색상/두께) 및
    저장/녹화/도움말 토글을 제어하는 매니저.
    """
    def __init__(self):
        # 드로잉 상태(blackboard에 반영)
        self.pen_color = (255, 255, 255)  # 기본 흰색
        self.thickness = 8
        self.last_msg = ""                # HUD 메시지(짧은 피드백)
        # 도움말/HUD 토글
        self.help_on = False

        # 녹화 상태
        self.is_recording = False
        self.writer = None
        self.rec_fps = 30  # 필요 시 조정
        self.rec_path_dir = "recordings"
        self.cap_path_dir = "captures"
        os.makedirs(self.rec_path_dir, exist_ok=True)
        os.makedirs(self.cap_path_dir, exist_ok=True)

    # ---- 유틸 ----
    def _ts(self) -> str:
        # 파일명 안전한 타임스탬프
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

    # ---- 외부에서 호출 ----
    def after_render(self, output_image):
        """
        렌더링 완료된 최종 프레임을 받아, 녹화 중이면 기록한다.
        (메인 루프에서 output_image가 만들어진 뒤 매 프레임 호출)
        """
        if self.is_recording and self.writer is not None:
            # VideoWriter는 BGR 3채널 uint8 기대
            frame = output_image
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            self.writer.write(frame)

    def apply_to_blackboard(self, blackboard):
        """
        현재 매니저 상태(pen_color, thickness)를 blackboard에 반영.
        (메인 루프에서 키 입력 처리 후, 한 번씩 호출해주면 된다)
        """
        blackboard.draw_color = self.pen_color
        blackboard.draw_thickness = self.thickness

    def handle_key(self, key, blackboard, current_frame_for_snapshot=None):
        """
        키 입력을 받아 내부 상태를 변경하고, 필요한 경우 부수효과(녹화/스냅샷)를 수행.
        - blackboard의 mode 전환은 기존 's' 키 로직과 충돌하지 않도록 별도 키 사용
        - 이미 main.py에 있는 키(A/D, ←/→, F, C, ↑/↓, Q 등)는 그대로 두고,
          여기서는 추가 기능을 담당한다.
        """
        if key == -1:
            return

        # ====== 추가 기능들 ======
        # 펜 색상 단축키
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

        # 펜 굵기 조절
        elif key in (ord('+'), ord('=')):  # 키보드 레이아웃 고려하여 '='도 함께
            self.thickness = min(self.thickness + 2, 60)
            self.last_msg = f"Thickness: {self.thickness}"
        elif key in (ord('-'), ord('_')):
            self.thickness = max(self.thickness - 2, 1)
            self.last_msg = f"Thickness: {self.thickness}"

        # 프리셋 1~5: 색+굵기 조합 (예시)
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

        # 녹화 토글: V
        elif key == ord('v'):
            # 프레임 크기: blackboard 해상도
            if not self.is_recording:
                self._start_recording(blackboard.width, blackboard.height)
            else:
                self._stop_recording()

        # 스냅샷: P
        elif key == ord('p'):
            if current_frame_for_snapshot is not None:
                self._save_snapshot(current_frame_for_snapshot)

        # 도움말: H
        elif key == ord('h'):
            self.help_on = not self.help_on
            self.last_msg = "Help ON" if self.help_on else "Help OFF"

        # 그 외: 여기서 처리 안 한 키는 main.py 기존 로직이 처리하도록 둔다.

    # 간단한 도움말 문자열
    def help_lines(self):
        return [
            "Shortcuts (KeyboardInputManager):",
            "  w/r/g/b/y : pen color (white/red/green/blue/yellow)",
            "  +/-       : pen thickness up/down",
            "  1..5      : presets",
            "  v         : start/stop recording (MP4)",
            "  p         : snapshot (PNG)",
            "  h         : toggle help",
            "",
            "Existing keys (main.py):",
            "  s         : toggle shape mode",
            "  c         : clear canvas",
            "  f         : open background file",
            "  a/d or ←/→ : prev/next page",
            "  ↑/↓       : zoom in/out",
            "  q         : quit",
        ]
