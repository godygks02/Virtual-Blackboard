# view_manager.py
import cv2
import numpy as np
from overlay_hud import draw_hud


class ViewManager:
    """
    - HUD 그리기
    - BG 에러 메시지 전달
    - 보기 모드(normal / pip):
        normal: 기존처럼 전체 화면
        pip: 사람은 우측 하단에 작은 창으로, 메인 화면은 '사람 없는 칠판'
    """

    def __init__(self):
        self.view_mode = "normal"   # "normal" or "pip"

    def toggle_mode(self):
        # 보기 모드 토글
        self.view_mode = "pip" if self.view_mode == "normal" else "normal"
        print(f"[VIEW] mode = {self.view_mode}")

    def compose(self, base_frame, blackboard, kb_manager):
        """
        - base_frame: VirtualBlackboard.update(frame)의 결과 (기존 최종 화면)
        - blackboard: last_combined_bg / last_frame 사용
        - 반환: 최종 디스플레이 프레임
        """

        # BG 에러 메시지 있으면 가져와서 HUD에 같이 넘김 (없으면 None)
        extra_msg = getattr(blackboard.bg_manager, "last_error", None)

        # 기본은 원래 최종 화면 위에 HUD
        frame_for_hud = base_frame

        # pip 모드일 때: 사람 없는 칠판 + PIP 카메라
        if self.view_mode == "pip":
            # 사람 없는 칠판 우선 (없으면 기존 최종 프레임 사용)
            if blackboard.last_combined_bg is not None:
                base = blackboard.last_combined_bg.copy()
            else:
                base = base_frame.copy()

            h, w = base.shape[:2]

            if blackboard.last_frame is not None:
                cam = blackboard.last_frame.copy()

                # 1) PIP 크기 설정 (화면 높이의 1/4 정도)
                pip_h = int(h * 0.25)
                pip_w = int(cam.shape[1] * (pip_h / cam.shape[0]))
                pip = cv2.resize(cam, (pip_w, pip_h))

                # 2) 동그라미 마스크 생성 (가운데 중심 원)
                mask = np.zeros((pip_h, pip_w), dtype=np.uint8)
                radius = min(pip_w, pip_h) // 2
                center = (pip_w // 2, pip_h // 2)
                cv2.circle(mask, center, radius, 255, -1)

                # 3) 원형으로 잘린 얼굴 PIP
                pip_fg = cv2.bitwise_and(pip, pip, mask=mask)

                # 4) 배치 위치: 우측 하단
                margin = 20
                y2 = h - margin
                y1 = y2 - pip_h
                x2 = w - margin
                x1 = x2 - pip_w

                # (만약 화면 중앙에 두고 싶으면)
                # cx, cy = w // 2, h // 2
                # x1 = cx - pip_w // 2
                # x2 = cx + pip_w // 2
                # y1 = cy - pip_h // 2
                # y2 = cy + pip_h // 2

                # 5) 배경 ROI에서 원 자리 비우기
                roi = base[y1:y2, x1:x2]
                mask_inv = cv2.bitwise_not(mask)  # 원 바깥 부분만 남는 마스크
                bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                # 6) 배경 + 얼굴 합성
                dst = cv2.add(bg, pip_fg)
                base[y1:y2, x1:x2] = dst

            frame_for_hud = base
        else:
            frame_for_hud = base_frame

        final = draw_hud(frame_for_hud, blackboard, kb_manager, extra_msg=extra_msg)
        return final
