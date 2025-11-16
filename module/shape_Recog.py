import cv2
import numpy as np
from collections import deque
import math  # Add math module for circle recognition


class ShapeRecognizer:
    """
    Manages hand-drawn shape data and converts/corrects closed areas into rectangles, triangles, or circles.
    """

    def __init__(
        self,
        history_len=500,
        min_contour_area=500,
        approx_epsilon=0.04,
        draw_color=(255, 255, 255),
        draw_thickness=8,
        erase_color=(0, 0, 0),
    ):

        self.current_drawing_pts = deque(maxlen=history_len)
        self.MIN_CONTOUR_AREA = min_contour_area
        self.APPROX_EPSILON = approx_epsilon
        self.prev_mode = "none"

        # Color and thickness
        self.draw_color = draw_color
        self.draw_thickness = draw_thickness
        self.erase_color = erase_color

        # Circle recognition threshold (ratio of contour area to bounding box area, if >= 0.75, it's a circle)
        self.CIRCLE_MATCH_THRESHOLD = 0.75

    # [ADD] Change shape color 251109
    def set_draw_color(self, color):
        self.draw_color = color

    def add_point(self, point):
        """Adds coordinates to the buffer when in drawing mode"""
        if point != (-1, -1):
            self.current_drawing_pts.append(point)

    def process_drawing(self, mode, canvas):
        """
        Processes shape recognition and canvas updates based on the current drawing mode.
        """
        if self.prev_mode == "draw" and mode in ("move", "none", "erase"):

            if len(self.current_drawing_pts) > 10:
                # Attempt shape correction (function name changed)
                success = self._recognize_and_draw_shape(canvas)

                # Clear buffer after processing
                self.current_drawing_pts.clear()
                self.prev_mode = mode
                return success

        self.prev_mode = mode
        return False

    def _recognize_and_draw_shape(self, canvas):
        """
        Finds a closed area based on saved coordinates and corrects it to a rectangle/triangle/circle.
        """

        # 1. Create temporary mask with drawing coordinates and find contours (maintain existing logic)
        pts = np.array(self.current_drawing_pts, dtype=np.int32)
        temp_mask = np.zeros_like(canvas[:, :, 0])
        cv2.polylines(temp_mask, [pts], isClosed=False, color=255, thickness=40)

        # Apply Closing morphological operation
        kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size through testing
        temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area < self.MIN_CONTOUR_AREA:
            return False

        # 2. Analyze and recognize shape
        shape_type = "unknown"

        # Calculate the perimeter of the contour and set epsilon as a ratio of the perimeter
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, self.APPROX_EPSILON * peri, True)

        if len(approx) == 3:
            # 3 vertices: Triangle
            shape_type = "triangle"
            x, y, w, h = cv2.boundingRect(approx)

        elif len(approx) == 4 and cv2.isContourConvex(approx):
            # 4 vertices: Rectangle (existing logic)
            shape_type = "rectangle"
            x, y, w, h = cv2.boundingRect(approx)

        else:
            # More than 4 vertices: Circle or other complex shape
            # Attempt circle recognition
            x, y, w, h = cv2.boundingRect(c)  # Bounding box of the original contour

            if w > 0 and h > 0:
                bounding_box_area = w * h
                # Judge circularity by the ratio of contour area to bounding box area
                area_ratio = area / bounding_box_area

                # If aspect ratio is close to 1 and area ratio is high, it's a circle
                aspect_ratio = w / h

                if (
                    area_ratio >= self.CIRCLE_MATCH_THRESHOLD
                    and 0.7 <= aspect_ratio <= 1.3  # Adjust value if needed (circle recognition range)
                ):
                    shape_type = "circle"
                    # Calculate center and radius of the circle
                    ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                    center = (int(cx), int(cy))
                    radius = int(radius)

        # 3. Draw corrected shape (including erasing)
        if shape_type != "unknown":

            pts = np.array(self.current_drawing_pts, dtype=np.int32)

            if len(pts) > 1:
                # Erase by redrawing the path with a black line.
                # Set thickness slightly larger than the drawing thickness to ensure it's fully erased.
                erase_thickness = self.draw_thickness + 10

                for i in range(1, len(pts)):
                    pt1 = tuple(pts[i - 1])
                    pt2 = tuple(pts[i])
                    cv2.line(
                        canvas,
                        pt1,
                        pt2,
                        self.erase_color,  # Black
                        erase_thickness,
                    )

            # 2) Draw corrected shape
            if shape_type == "circle":
                # Draw corrected circle
                cv2.circle(canvas, center, radius, self.draw_color, self.draw_thickness)

            elif shape_type in ("rectangle", "triangle"):
                # Draw corrected rectangle or triangle outline (using approx)
                cv2.polylines(
                    canvas,
                    [approx],
                    isClosed=True,
                    color=self.draw_color,
                    thickness=self.draw_thickness,
                )

            print(f"Shape recognized and corrected: {shape_type}")
            return True

        return False
