---
title: OAK-D Pro에서 RGB와 Depth의 시야각과 픽셀 맞추기 (TurtleBot4, Jazzy)
date: 2026-10-04
status: draft
question: TurtleBot4의 OAK-D Pro에서 RGB 박스 좌표로 Depth를 읽으려면 무엇을 설정해야 하고, 어떻게 확인하는가?
---

# OAK-D Pro에서 RGB와 Depth의 시야각과 픽셀 맞추기

## 결론 (한 문장)
드라이버 기본값이 Depth를 RGB 카메라에 정렬하고 크기도 RGB(1280x720)에 맞추므로, `oakd_pro.yaml`에서 `camera.i_pipeline_type: RGBD` **한 줄**만 바꾸면 된다. **RGB를 480P로 낮추는 방법은 OAK-D Pro(IMX378)에서는 맞지 않는다.** 정렬 여부는 camera_info 비교가 아니라 두 영상을 겹쳐서 확인한다.

## 근거

| 주장 | 값 / 내용 | 출처 | 확인일 | 구분 |
|---|---|---|---|---|
| 슬라이드의 과제 | "EXERCISE: ACHIEVE ALIGNED FOV AND DIMENSION FOR BOTH DEPTH AND RGB", "Are Depth and RGB pixel aligned?" | 강의 자료 Day2 `Day2_new_v2_jazzy.pdf` | 2026-10-04 | 인용 |
| 카메라 | RGB IMX378(자동 초점), 흑백 스테레오 OV9282 | [1] | 2026-10-03 | 인용 |
| 센서 전체 시야각 | IMX378 가로 66° x 세로 54°, OV9282 가로 80° x 세로 55° | [1] | 2026-10-03 | 인용 |
| IMX378 모드 | 12MP(전체), 4K(자름), 1080P(4K로 자른 뒤 비닝). **480P 모드는 없다** | [2] | 2026-10-03 | 인용 |
| 지금 RGB 1280x720의 시야각 | 가로 63.7° x 세로 38.5° (fx = 1030.28, 로봇 3번 camera_info에서) | 2·atan(640/1030.28), 2·atan(360/1030.28) | 2026-10-03 | 추정 |
| 1280x720은 자른 것이 아니라 줄인 것 | 1080P를 2/3로 축소해서 시야각이 그대로다. `i_width`/`i_height`만 바꾸면 가운데를 잘라 시야각이 좁아진다 | [3][4] | 2026-10-03 | 인용 |
| 정렬 기본값 | `stereo.i_align_depth` 기본 true, 기준 카메라 CAM_A(RGB). Depth 크기는 `stereo.i_width`/`i_height`를 따로 주지 않으면 `rgb.i_width`/`i_height`를 따른다 | [5] `stereo_param_handler.cpp` | 2026-10-03 | 인용 |
| TurtleBot4 설정 | `oakd_pro.yaml`에 `stereo:` 설정이 없어 위 기본값이 그대로 쓰인다 | [6] | 2026-10-03 | 인용 |
| 정렬했을 때 camera_info | Depth의 K가 RGB와 같고(같은 크기일 때), frame_id가 RGB 광학 프레임이 된다 | [5][7] | 2026-10-03 | 인용 |
| Depth 최소 거리 | 흑백 1280 폭(드라이버 기본 720P), 기준선 7.5 cm, 시차 최대 95 px이면 약 0.60 m | fx = 640/tan 40° = 763 px, 763 × 0.075 / 95 | 2026-10-04 | 추정 |
| Depth 원본 대역폭 | 1280x720 16비트 = 1.84 MB/프레임, 30 fps면 약 55 MB/s | 계산 | 2026-10-03 | 추정 |

## 비교

| 방법 | 내용 | 우리 상황에 맞나 |
|---|---|---|
| A. RGBD 한 줄 (권장) | `camera.i_pipeline_type: RGBD`. Depth가 RGB에 정렬되고 RGB와 같은 1280x720이 된다. 박스 좌표를 그대로 쓴다 | 맞다 |
| A'. A + Depth만 작게 | A에서 `stereo.i_width`/`i_height`를 512x288이나 768x432처럼 16:9, 16의 배수로 준다. 같은 비율이라 좌표를 비율로 옮기면 된다 | 대역폭이 모자라면 |
| B. RGB를 480P로 | 슬라이드 예시 설정으로 보인다. IMX378에는 480P 모드가 없어 드라이버가 1080P로 바꾸고, 640x480은 가운데를 잘라 시야각이 줄어든다(가로 약 34.5°) | 맞지 않는다 [추정: 슬라이드는 OAK-D Lite 기준일 수 있다] |

Depth 값은 박스 중심 한 픽셀이 아니라 작은 창(예: 5x5)의 중앙값으로 읽고, 0과 NaN(스테레오가 못 맞춘 픽셀)은 뺀다. 팀 코드가 이미 이렇게 한다.

## 확인 방법
1. `ros2 param get /robot3/oakd stereo.i_align_depth`가 True인지 본다 [5].
2. RGB와 Depth의 camera_info를 비교한다(크기, K, frame_id). **이것은 설정 일관성 점검일 뿐이다.** 드라이버가 두 camera_info를 같은 보정값에서 계산하기 때문에, 정렬하도록 설정되어 있으면 실제 영상이 어긋나도 K는 같게 나온다.
3. **실물 증거는 겹쳐 보기다:** Depth를 색으로 칠해 RGB 위에 반투명하게 겹친다. 상자를 1.0, 1.5, 2.0 m에 두고 가운데와 모서리에서 테두리가 맞는지 본다. 0.5 m는 최소 거리보다 가깝다. 합격선(모서리 px, 거리 cm)은 측정 전에 정한다. 거리마다 어긋남이 다르면 정렬이 안 된 것이고, 모서리로 갈수록 커지면 렌즈 왜곡이다(Depth만 왜곡을 보정한다) [8].
4. Luxonis 예제 `rgb_depth_aligned`는 depthai로 장치를 직접 열어서, 드라이버가 장치를 잡고 있는 로봇에서는 같이 돌릴 수 없다. PC에서 두 토픽을 구독해 겹친다 [9].

## 한계와 미확인
- 로봇에 설치된 드라이버 버전은 확인하지 않았다 [미확인].
- Depth 출력 크기도 16의 배수여야 하는지는 문서가 RGB 쪽만 말한다 [미확인].
- 로봇의 Pi CPU와 Wi-Fi가 RGB와 Depth를 함께 버티는지는 재지 않았다 [미확인]. RGB만으로도 설정 30 fps에 실제 19~21 Hz다.
- 왜곡 때문에 모서리에 남는 어긋남의 크기는 모른다.

## 다음 단계
- [ ] 로봇 설정 변경은 팀 저장소의 절차(백업, 해시 확인, 사용자 승인)대로 하고, 겹쳐 보기로 실물 정렬을 확인한다.

## 출처
1. Luxonis, OAK-D Pro, https://docs.luxonis.com/hardware/products/OAK-D%20Pro, 확인일 2026-10-03
2. Luxonis, IMX378, https://docs.luxonis.com/hardware/sensors/IMX378, 확인일 2026-10-03
3. Luxonis, depthai-ros driver, https://docs.luxonis.com/software/ros/depthai-ros/driver, 확인일 2026-10-03
4. Luxonis, ColorCamera node, https://docs.luxonis.com/software/depthai-components/nodes/color_camera, 확인일 2026-10-03
5. luxonis/depthai-ros (jazzy), `depthai_ros_driver/src/param_handlers/stereo_param_handler.cpp`, https://github.com/luxonis/depthai-ros/blob/jazzy/depthai_ros_driver/src/param_handlers/stereo_param_handler.cpp, 확인일 2026-10-03
6. turtlebot/turtlebot4_robot (jazzy), `turtlebot4_bringup/config/oakd_pro.yaml`, https://github.com/turtlebot/turtlebot4_robot/blob/jazzy/turtlebot4_bringup/config/oakd_pro.yaml, 확인일 2026-10-03
7. luxonis/depthai-ros (jazzy), `depthai_bridge/src/ImageConverter.cpp`, https://github.com/luxonis/depthai-ros/blob/jazzy/depthai_bridge/src/ImageConverter.cpp, 확인일 2026-10-03
8. Luxonis, RGB-D, https://docs.luxonis.com/software/perception/rgb-d, 확인일 2026-10-03
9. Luxonis, rgb_depth_aligned example, https://docs.luxonis.com/software/depthai/examples/rgb_depth_aligned, 확인일 2026-10-03
