import tkinter as tk
from tkinter import filedialog

def file_select_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select PDF/Image for Background",
        filetypes=(
            ("PDF and Image Files", "*.pdf *.png *.jpg *.jpeg"),
            ("PDF Files", "*.pdf"),
            ("Image Files", "*.png *.jpg *.jpeg")
        )
    )

    return file_path
