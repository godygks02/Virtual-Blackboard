import fitz
print(fitz.__doc__)

import fitz  # PyMuPDF

# PDF 파일 열기
pdf_path = r"Lecture 6.pdf"
doc = fitz.open(pdf_path)

print(f"총 페이지 수: {len(doc)}")

import numpy as np
import cv2
page = doc.load_page(0)  # 0은 첫 페이지 (index 기반)
print(page)

pix = page.get_pixmap(dpi=150)  # DPI = 해상도 (높을수록 선명)
pix.save("page1.png")           # 파일로 저장

img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

# RGBA → BGR (OpenCV 호환)
if pix.n == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imshow("PDF Page", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
