# Virtual Blackboard

An interactive, gesture-based virtual blackboard project using OpenCV and MediaPipe. You can draw in real-time using hand gestures in front of your webcam, use lecture materials (images/PDFs) as a background, and conduct lectures.

##  Features

- ** Real-time Hand Gesture Recognition**:
  - **Mode Switching**: Automatically switches between 'Draw', 'Erase', and 'Cursor Move' modes based on your finger gestures.
  - **Coordinate Smoothing**: Applies a moving average filter for smoother cursor movement.

- ** Advanced Drawing Functions**:
  - **Shape Correction**: Recognizes hand-drawn shapes (circles, rectangles, triangles) and automatically corrects them into neat figures.
  - **Versatile Pen Settings**: Easily change color and thickness with hotkeys and supports presets.

- ** Dynamic Background Management**:
  - **Multiple Format Support**: Set solid colors, images (JPG, PNG), or multi-page PDF files as your background.
  - **Background Control**: Freely zoom in/out and pan the background using your mouse wheel and drag.
  - **Page Navigation**: Turn pages of a PDF background using the keyboard.

- ** User Segmentation (Background Removal)**:
  - Utilizes MediaPipe Selfie Segmentation to separate the user (foreground) from the background in real-time.
  - The user appears to be 'in front of' the blackboard, preventing drawings from overlapping with the user.

- ** Convenience Features**:
  - **Multiple View Modes**: Supports a normal mode and a PIP (Picture-in-Picture) mode that displays the presenter's face in a small window at the bottom right.
  - **Status HUD**: A Heads-Up Display shows the current pen settings, mode, page information, and more.
  - **Recording and Snapshots**: Record the current screen as a video (MP4) or capture it as an image (PNG).
  - **Help Panel**: Instantly view all keyboard shortcuts on-screen.

## Tech Stack

- **`OpenCV`**: Webcam I/O, image processing, and rendering.
- **`MediaPipe`**: Hand landmark detection (Hands) and user segmentation (Selfie Segmentation).
- **`NumPy`**: High-performance image array (ndarray) manipulation.
- **`PyMuPDF` (fitz)**: PDF file rendering and page management.
- **`cvzone`**: A wrapper to simplify the use of MediaPipe models.

##  Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/godygks02/Virtual-Blackboard.git
cd Virtual-Blackboard
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 4. Run the Program
```bash
python main.py
```

##  Controls

### Mouse Controls
- **Mouse Wheel (Up/Down)**: Zoom In / Zoom Out background.
- **Left-click and Drag**: Pan background.
- **Left-click Double-click**: Reset background zoom and position.

### Keyboard Shortcuts
| Key | Function |
|---|---|
| **q** | **Quit Program** |
| **c** | **Clear** entire canvas |
| **s** | Toggle **Shape Recognition Mode** |
| **z** | Toggle **View Mode** (Normal ↔ PIP) |
| **h** | Toggle **Help Panel** |
| **`** | Toggle entire **HUD** |
| **t** | Toggle **Hand Tracking (Drawing)** |
| **u** | Toggle **User Mask (Background Removal)** |
| | |
| **f** | Open background file (Image/PDF) |
| **x** | Close background file and switch to a black blackboard |
| **←** / **a** | PDF Previous Page |
| **→** / **d** | PDF Next Page |
| **↑** | Zoom In background |
| **↓** | Zoom Out background |
| | |
| **w, r, g, b, y** | Change pen color (White, Red, Green, Blue, Yellow) |
| **+** / **-** | Adjust pen thickness |
| **1 ~ 5** | Pen setting presets |
| | |
| **v** | Start/Stop **Video Recording** |
| **p** | Save **Snapshot** of the current screen |

##  Rendering Architecture

This project composites three virtual layers in real-time to create the final output.

```
[Final Screen]
    ↑
<Composite with Drawing>
    ↑
+----------------------+      +----------------------+
|   (Layer 3) User     |      |   (Layer 2) Drawing  |
| (Segmented from Cam) |  +   |      (Canvas)        |
+----------------------+      +----------------------+
    ↑
<Composite with Background>
    ↑
+----------------------+
|   (Layer 1) Background |
| (PDF/Image/Solid)    |
+----------------------+
```
1.  The **User Layer (Layer 3)** is composited on top of the **Background Layer (Layer 1)**.
2.  The **Drawing Layer (Layer 2)** is then composited on top of that result, creating the effect of the user being 'behind' the drawing.

##  Module Descriptions

- **`main.py`**:
  - The **control tower** that initializes all modules and runs the main loop.
  - It manages webcam frame processing, keyboard input detection, and final screen rendering.

- **`handTracker.py`**:
  - Tracks the 21 landmarks of a single hand using `MediaPipe Hands`.
  - Determines the 'draw', 'erase', and 'move' states by calculating the distance between the thumb and index fingertips.

- **`BackGroundManager.py`**:
  - Manages the background layer. It renders PDFs page by page via `PyMuPDF`, loads image files, and handles zoom/pan states.

- **`UserMaskManager.py`**:
  - Runs the `MediaPipe Selfie Segmentation` model via the `cvzone` library.
  - Creates a mask that segments only the user area from the webcam frame.

- **`shape_Recog.py`**:
  - Stores the trajectory of a user's drawing in a buffer. When the drawing action finishes, it analyzes the path.
  - It finds a closed contour, recognizes it as a triangle, rectangle, or circle, and draws the corrected shape on the canvas.

- **`keyboard_input.py`**:
  - A state manager that handles various keyboard inputs for pen settings, feature toggles, and media controls (recording/capture).

- **`overlay_hud.py`**:
  - Draws the HUD that visually displays the current program state (mode, pen, page, etc.).
  - Manages the text and layout for the help panel.

- **`view_manager.py`**:
  - Manages and switches the screen composition between normal and PIP modes.

- **`utils.py`**:
  - Opens a GUI file dialog using `tkinter` to allow the user to select a background file.
