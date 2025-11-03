import os
import cv2
import numpy as np
import fitz  # PyMuPDF (for PDF)

class BackgroundManager:
    def __init__(self, width, height, dpi=150):
        self.width = width
        self.height = height
        self.dpi = dpi

        # 상태 변수
        self.mode = "solid"     # solid / image / pdf
        self.background = np.zeros((height, width, 3), np.uint8)
        self.color = (0, 0, 0)

        # PDF 전용
        self.doc = None
        self.page_index = 0

        # 인터랙션
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.drag_start = (0, 0)
        self.possible_prev_page = None

    # =======================================================
    #  배경 로딩
    # =======================================================
    def add_background(self, source=None, color=(0, 0, 0)):
        if source is None:
            self.mode = "solid"
            self.background[:] = color
            print(f"[BG] 단색({color}) 설정")
            return

        ext = os.path.splitext(source)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            self.mode = "image"
            img = cv2.imread(source)
            if img is not None:
                self.background = cv2.resize(img, (self.width, self.height))
                self.background = cv2.cvtColor(self.background, cv2.COLOR_RGB2BGR)
                print(f"[BG] 이미지 '{source}' 로드 완료")
            else:
                print(f"[BG] 이미지 '{source}' 불러오기 실패")

        elif ext == ".pdf":
            self.mode = "pdf"
            self.doc = fitz.open(source)
            self.page_index = 0
            print(f"[BG] PDF '{source}' 로드 ({len(self.doc)}페이지)")
            self.background = self._render_pdf_page(self.page_index)

        else:
            print(f"[BG] 지원되지 않는 파일 형식: {ext}")

    # =======================================================
    #  PDF 렌더링
    # =======================================================
    def _render_pdf_page(self, index):
        if not self.doc:
            return self.background
        try:
            page = self.doc.load_page(index)
        except:
            print(F"INDEX{index} PAGE IS NOT EXIST")
            return self.possible_prev_page
        pix = page.get_pixmap(dpi=self.dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        self.possible_prev_page = cv2.resize(img, (self.width, self.height))
        self.possible_prev_page = cv2.cvtColor(self.possible_prev_page, cv2.COLOR_RGB2BGR)
        return self.possible_prev_page

    def next_page(self):
        if self.doc and self.page_index < len(self.doc) - 1:
            self.page_index += 1
            self.background = self._render_pdf_page(self.page_index)
            print(f"[PDF] {self.page_index + 1}/{len(self.doc)} 페이지")

    def prev_page(self):
        if self.doc and self.page_index > 0:
            self.page_index -= 1
            self.background = self._render_pdf_page(self.page_index)
            print(f"[PDF] {self.page_index + 1}/{len(self.doc)} 페이지")

    # =======================================================
    # 줌 & 드래그
    # =======================================================
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.zoom = min(self.zoom * 1.1, 5.0)
            else:
                self.zoom = max(self.zoom * 0.9, 0.3)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx, dy = x - self.drag_start[0], y - self.drag_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.drag_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.zoom = 1.0
            self.offset_x = self.offset_y = 0
            print("[BG] 줌/위치 리셋")

    # =======================================================
    # 뷰 반환 (줌 + 이동 적용)
    # =======================================================
    def get_view(self):
        img = self.background.copy()
        scaled = cv2.resize(img, None, fx=self.zoom, fy=self.zoom)
        sh, sw = scaled.shape[:2]

        cx = self.width // 2 - sw // 2 + self.offset_x
        cy = self.height // 2 - sh // 2 + self.offset_y

        result = np.zeros((self.height, self.width, 3), np.uint8)
        x1, y1 = max(cx, 0), max(cy, 0)
        x2, y2 = min(cx + sw, self.width), min(cy + sh, self.height)
        src_x1, src_y1 = max(0, -cx), max(0, -cy)
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)

        result[y1:y2, x1:x2] = scaled[src_y1:src_y2, src_x1:src_x2]
        return result
