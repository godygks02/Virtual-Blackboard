# Virtual-Blackboard
현재 **손 인식(Hand Tracking)**과 **배경 제거(Background Removal)**의 두 가지 핵심 모듈이 구현

---

##  Setup

libraries

```bash
# 1. OpenCV, MediaPipe
pip install opencv-python
pip install mediapipe

# 2. 배경 제거 모듈 (BackGroundModule.py)을 위한 필수 라이브러리
# cvzone은 1.5.6 버전을 사용
pip install cvzone==1.5.6
```
---

## Module Descriptions

### handTracking.py

- **설명:** `mediapipe.solutions.hands`를 기반으로 사용자의 손을 감지하고 추적하는 모듈
    - 웹캠 프레임에서 손의 21개 랜드마크를 실시간으로 감지
    - (현재 로직) **엄지(4번)**와 **검지(8번) 끝** 사이의 거리를 계산
    - 이 거리를 기반으로 사용자의 제스처(예: '그리기', '지우기', '이동')를 판단할 수 있는 상태 값과 좌표를 반환

### BackGroundModule.py

- **설명:** 웹캠 프레임에서 사용자(사람) 영역만 정확히 분리하고 나머지 배경을 제거.
- **주요 기능:**
    - `cvzone.SelfieSegmentationModule` (1.5.6 버전)을 사용
    - 입력된 프레임(원본 영상)에서 배경을 제거하고 사용자만 남긴 **세그멘테이션 마스크(Mask)** 또는 **결과 이미지**를 반환

### main.py

- **설명:** handTracker.py와 BackGroundModule.py를 통합하여 가상 칠판의 모든 기능을 실행하는 메인 컨트롤 타워
- 모든 레이어 관리와 렌더링을 총괄

- **주요 기능:**
#### 3-Layer 렌더링 아키텍처
3개의 가상 레이어를 합성하여 최종 화면 생성

- [Layer 1] 배경 레이어 (Back): self.background

   - BackGroundModule을 통해 생성된 정적 레이어
   - (현재 검은색 (0, 0, 0) 칠판으로 설정됨)

- [Layer 2] 드로잉 레이어 (Middle): self.canvas

   - handTracker의 입력에 따라 그림이 그려지는 레이어

   - (현재 검은색 (0, 0, 0) 캔버스, 흰색 (255, 255, 255) 잉크)

- [Layer 3] 사용자 레이어 (Front): user_part (원본 프레임에서 추출)

   - BackGroundModule이 생성한 user_mask를 이용해 원본 웹캠(frame)에서 사용자 부분만 오려낸 동적 레이어

---

## 실행 및 사용법

프로젝트를 실행하려면 터미널에서 main.py를 직접 실행
```bash
python main.py
```

## 키보드 조작

- q: 프로그램 종료
- c: 캔버스 지우기 (드로잉 레이어를 검은색으로 초기화)

---

