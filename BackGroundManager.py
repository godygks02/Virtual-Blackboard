import os
import cv2
import numpy as np
import fitz  # PyMuPDF (for PDF)

class BackgroundManager:
    def __init__(self, width, height, dpi=150):
        self.width = width
        self.height = height
        self.dpi = dpi

        # State variables
        self.mode = "solid"     # solid / image / pdf
        self.background = np.zeros((height, width, 3), np.uint8)
        self.color = (0, 0, 0)

        # PDF specific
        self.doc = None
        self.page_index = 0

        # Interaction
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.drag_start = (0, 0)
        self.possible_prev_page = None

        # Error message to display on HUD, etc.
        self.last_error = ""

    # =======================================================
    #  Background Loading
    # =======================================================
    def add_background(self, source=None, color=(0, 0, 0)):
        # Clear previous error
        self.last_error = ""
        self.color = color

        if source is None:
            self.mode = "solid"
            self.background[:] = color
            self.color = color
        
             # Reset PDF related states
            self.doc = None
            self.page_index = 0

            print(f"[BG] Solid color({color}) set and page info reset")
            return

        ext = os.path.splitext(source)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            self.mode = "image"
            img = cv2.imread(source)
            if img is not None:
                self.background = cv2.resize(img, (self.width, self.height))
                self.background = cv2.cvtColor(self.background, cv2.COLOR_RGB2BGR)
                print(f"[BG] Image '{source}' loaded successfully")
            else:
                msg = f"[BG] Failed to load image '{source}'"
                print(msg)
                self.mode = "solid"
                self.background[:] = color
                self.last_error = msg

        elif ext == ".pdf":
            self.mode = "pdf"
            try:
                self.doc = fitz.open(source)
                self.page_index = 0
                print(f"[BG] PDF '{source}' loaded ({len(self.doc)} pages)")
                self.background = self._render_pdf_page(self.page_index)
            except Exception as e:
                msg = f"[BG] Failed to load PDF '{source}': {e}"
                print(msg)
                self.doc = None
                self.mode = "solid"
                self.background[:] = color
                self.last_error = msg
        else:
            msg = f"[BG] Unsupported file format: {ext}"
            print(msg)
            self.last_error = msg

    # =======================================================
    #  PDF Rendering
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

    @property
    def current_page(self):
        return self.page_index

    @property
    def total_pages(self):
        return len(self.doc) if self.doc is not None else 1

    def next_page(self):
        if self.doc and self.page_index < len(self.doc) - 1:
            self.page_index += 1
            self.background = self._render_pdf_page(self.page_index)
            print(f"[PDF] Page {self.page_index + 1}/{len(self.doc)}")

    def prev_page(self):
        if self.doc and self.page_index > 0:
            self.page_index -= 1
            self.background = self._render_pdf_page(self.page_index)
            print(f"[PDF] Page {self.page_index + 1}/{len(self.doc)}")

    # =======================================================
    # Zoom & Drag
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
            print("[BG] Zoom/Position reset")

    # =======================================================
    # Return view (zoom + pan applied)
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
