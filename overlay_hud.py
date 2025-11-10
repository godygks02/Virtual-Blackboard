# overlay_hud.py
import cv2
import numpy as np

def _draw_text(img, text, org, color=(255,255,255), scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_hud(frame_bgr, blackboard, kb_manager, extra_msg=None):
    """
    Draws a status HUD in the top-left corner of the screen.
    - Displays mode, color, thickness, zoom, page, and rec status.
    - If kb_manager.help_on is true, displays a simple help panel.
    """
    # HUD ON/OFF Toggle
    # If hud_on is False (toggled by '`' key), return the original frame without drawing anything.
    if not kb_manager.hud_on:
        return frame_bgr
    
    img = frame_bgr
    h, w = img.shape[:2]

    # Translucent panel
    panel_w, panel_h = min(710, w-40), 110
    overlay = img.copy()
    cv2.rectangle(overlay, (20, 20), (20+panel_w, 20+panel_h), (0,0,0), -1)
    img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    # ===== Recording ON/OFF indicator =====
    rec = "ON" if kb_manager.is_recording else "OFF"   # ← Changed
    trk = "ON" if kb_manager.drawing_enabled else "OFF"
    usr = "ON" if kb_manager.user_mask_enabled else "OFF"

    mode = blackboard.drawing_mode
    color = kb_manager.pen_color
    thick = kb_manager.thickness
    zoom = getattr(blackboard.bg_manager, "zoom", 1.0)

    # Display page only when PDF is active
    if blackboard.bg_manager.doc:
        total_pages = len(blackboard.bg_manager.doc)
        page = blackboard.bg_manager.page_index + 1
    else:
        total_pages = 1
        page = 0  # Display as 0/1 when PDF is off

    # ===== Dynamically calculate x-coordinate of the right column by measuring string width =====
    # Row1: "Mode: ..." | "REC: ... | TRK: ... | USR: ..."
    row1_left  = f"Mode: {mode}"
    row1_right = f"REC: {rec}  TRK: {trk}  USR: {usr}"
    (left_w1, _), _ = cv2.getTextSize(row1_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    gap = 20  # Gap between left/right columns
    x_left = 30
    y1 = 50
    _draw_text(img, row1_left, (x_left, y1))
    _draw_text(img, row1_right, (x_left + left_w1 + gap, y1))  # ← Dynamic x calculation to avoid overlap

    # Row2: "Pen: (..).. Thick: .." | "Zoom: .. Page: .."
    row2_left  = f"Pen: {color}  Thick: {thick}"
    row2_right = f"Zoom: {zoom:.2f}  Page: {page}/{total_pages}"
    (left_w2, _), _ = cv2.getTextSize(row2_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y2 = 80
    _draw_text(img, row2_left, (x_left, y2))
    _draw_text(img, row2_right, (x_left + left_w2 + gap, y2))  # ← Dynamic x instead of fixed x(260)

    # Short message (last action)
    if kb_manager.last_msg:
        _draw_text(img, kb_manager.last_msg, (30, 108), (0,255,255), 0.6, 2)

    # Help panel
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

    # Additional message
    if extra_msg:
        _draw_text(img, extra_msg, (30, h-30), (0,255,0), 0.7, 2)

    return img
