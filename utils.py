import tkinter as tk
from tkinter import filedialog

def file_select_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="배경으로 사용할 PDF/이미지 선택",
        filetypes=(
            ("PDF 및 이미지 파일", "*.pdf *.png *.jpg *.jpeg"),
            ("PDF 파일", "*.pdf"),
            ("이미지 파일", "*.png *.jpg *.jpeg")
        )
    )

    return file_path