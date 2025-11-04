# overlay_hud.py
import cv2
import numpy as np

def _draw_text(img, text, org, color=(255,255,255), scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_hud(frame_bgr, blackboard, kb_manager, extra_msg=None):
    """
    화면 좌측 상단에 상태 HUD를 그려준다.
    - mode, color, thickness, zoom, page, rec 상태 표시
    - kb_manager.help_on이면 간단한 도움말 패널 표시
    """
    img = frame_bgr
    h, w = img.shape[:2]

    # 반투명 패널 
    panel_w, panel_h = min(480, w-40), 110
    overlay = img.copy()
    cv2.rectangle(overlay, (20, 20), (20+panel_w, 20+panel_h), (0,0,0), -1)
    img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    # ===== 녹화 ON/OFF 표기 =====
    rec = "ON" if kb_manager.is_recording else "OFF"   # ← 변경
    mode = blackboard.drawing_mode
    color = kb_manager.pen_color
    thick = kb_manager.thickness
    zoom = getattr(blackboard.bg_manager, "zoom", 1.0)
    page = getattr(blackboard.bg_manager, "current_page", 0) + 1

    # ===== 문자열 폭 측정해 오른쪽 열 x좌표를 동적으로 산정 =====
    # Row1: "Mode: ..." | "REC: ..."
    row1_left  = f"Mode: {mode}"
    row1_right = f"REC: {rec}"
    (left_w1, _), _ = cv2.getTextSize(row1_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    gap = 20  # 좌/우 열 간격
    x_left = 30
    y1 = 50
    _draw_text(img, row1_left, (x_left, y1))
    _draw_text(img, row1_right, (x_left + left_w1 + gap, y1))  # ← 겹치지 않도록 동적 x 계산

    # Row2: "Pen: (..).. Thick: .." | "Zoom: .. Page: .."
    row2_left  = f"Pen: {color}   Thick: {thick}"
    row2_right = f"Zoom: {zoom:.2f}   Page: {page}"
    (left_w2, _), _ = cv2.getTextSize(row2_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y2 = 80
    _draw_text(img, row2_left, (x_left, y2))
    _draw_text(img, row2_right, (x_left + left_w2 + gap, y2))  # ← 고정 x(260) 대신 동적 x

    # 짧은 메시지(최근 액션)
    if kb_manager.last_msg:
        _draw_text(img, kb_manager.last_msg, (30, 108), (0,255,255), 0.6, 2)

    # 도움말 패널
    if kb_manager.help_on:
        lines = kb_manager.help_lines()
        pad = 10
        box_w = 530
        box_h = 22*(len(lines)+1)
        y0 = 20 + panel_h + 10
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (20, y0), (20+box_w, y0+box_h), (30,30,30), -1)
        img[:] = cv2.addWeighted(overlay2, 0.70, img, 0.30, 0)
        y = y0 + 28
        for ln in lines:
            _draw_text(img, ln, (30, y), (255,255,255), 0.6, 1)
            y += 22

    # 추가 메시지
    if extra_msg:
        _draw_text(img, extra_msg, (30, h-30), (0,255,0), 0.7, 2)

    return img
