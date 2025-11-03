import cv2
import numpy as np
from collections import deque
import math  # 원 인식을 위해 math 모듈 추가


class ShapeRecognizer:
    """
    손으로 그린 도형 데이터를 관리하고 닫힌 영역을 사각형, 삼각형, 원으로 변환 (보정)
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

        # 색상 및 두께
        self.draw_color = draw_color
        self.draw_thickness = draw_thickness
        self.erase_color = erase_color

        # 원 인식 임계값 (컨투어 면적 / 바운딩 사각형 면적 비율, 0.75 이상이면 원으로 판단)
        self.CIRCLE_MATCH_THRESHOLD = 0.75

    def add_point(self, point):
        """그리기 모드일 때 좌표를 버퍼에 추가"""
        if point != (-1, -1):
            self.current_drawing_pts.append(point)

    def process_drawing(self, mode, canvas):
        """
        현재 드로잉 모드에 따라 도형 인식 및 캔버스 업데이트를 처리
        """
        if self.prev_mode == "draw" and mode in ("move", "none", "erase"):

            if len(self.current_drawing_pts) > 10:
                # 도형 보정 시도 (함수 이름 변경)
                success = self._recognize_and_draw_shape(canvas)

                # 처리 후 버퍼 초기화
                self.current_drawing_pts.clear()
                self.prev_mode = mode
                return success

        self.prev_mode = mode
        return False

    def _recognize_and_draw_shape(self, canvas):
        """
        저장된 좌표를 기반으로 닫힌 영역을 찾고 사각형/삼각형/원으로 보정
        """

        # 1. 드로잉 좌표로 임시 마스크 생성 및 컨투어 찾기 (기존 로직 유지)
        pts = np.array(self.current_drawing_pts, dtype=np.int32)
        temp_mask = np.zeros_like(canvas[:, :, 0])
        cv2.polylines(temp_mask, [pts], isClosed=False, color=255, thickness=30)

        contours, _ = cv2.findContours(
            temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area < self.MIN_CONTOUR_AREA:
            return False

        # 2. 도형 분석 및 인식
        shape_type = "unknown"

        # 컨투어의 길이를 계산하고, 이 길이의 APPROX_EPSILON 비율만큼 엡실론을 설정
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, self.APPROX_EPSILON * peri, True)

        if len(approx) == 3:
            # 꼭짓점이 3개: 삼각형
            shape_type = "triangle"
            x, y, w, h = cv2.boundingRect(approx)

        elif len(approx) == 4 and cv2.isContourConvex(approx):
            # 꼭짓점이 4개: 사각형 (기존 로직)
            shape_type = "rectangle"
            x, y, w, h = cv2.boundingRect(approx)

        else:
            # 4개보다 많은 꼭짓점: 원 또는 다른 복잡한 도형
            # 원 인식 시도
            x, y, w, h = cv2.boundingRect(c)  # 원본 컨투어의 바운딩 박스

            if w > 0 and h > 0:
                bounding_box_area = w * h
                # 컨투어 면적 / 바운딩 사각형 면적 비율로 원형 판단
                area_ratio = area / bounding_box_area

                # 가로세로 비율이 1에 가깝고 면적 비율이 높으면 원으로 판단
                aspect_ratio = w / h

                if (
                    area_ratio >= self.CIRCLE_MATCH_THRESHOLD
                    and 0.8 <= aspect_ratio <= 1.2
                ):
                    shape_type = "circle"
                    # 원의 중심과 반지름 계산
                    ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                    center = (int(cx), int(cy))
                    radius = int(radius)

        # 3. 보정된 도형 그리기 (지우기 포함)
        if shape_type != "unknown":

            # # 1) 잔상 지우기: 도형의 바운딩 박스 영역을 지우기 색상으로 채움
            # # (원의 경우 minEnclosingCircle의 결과를 사용하여 안전하게 지워야 함)

            # if shape_type == "circle":
            #     # 원의 바운딩 박스를 사용 (원의 중심과 지름 기반)
            #     x_c, y_c = center
            #     radius_c = radius
            #     cv2.rectangle(
            #         canvas,
            #         (x_c - radius_c - 20, y_c - radius_c - 20),
            #         (x_c + radius_c + 20, y_c + radius_c + 20),
            #         self.erase_color,
            #         cv2.FILLED,
            #     )
            # else:
            #     # 사각형 또는 삼각형의 바운딩 박스 사용
            #     cv2.rectangle(
            #         canvas,
            #         (x - 20, y - 20),
            #         (x + w + 20, y + h + 20),
            #         self.erase_color,
            #         cv2.FILLED,
            #     )

            # 2) 보정된 도형 그리기
            if shape_type == "circle":
                # 보정된 원 그리기
                cv2.circle(canvas, center, radius, self.draw_color, self.draw_thickness)

            elif shape_type in ("rectangle", "triangle"):
                # 보정된 사각형 또는 삼각형 외곽선 그리기 (approx 사용)
                cv2.polylines(
                    canvas,
                    [approx],
                    isClosed=True,
                    color=self.draw_color,
                    thickness=self.draw_thickness,
                )

            print(f"도형 인식 및 보정 완료: {shape_type}")
            return True

        return False
