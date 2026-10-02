---
title: 최신 객체 탐지 모델 조사 (GitHub 기준, 2026-10-02)
date: 2026-10-02
status: draft   # draft | reviewed | final
question: RC카를 탐지하는 우리 프로젝트의 비교 실험에, 2026-10 시점의 최신 탐지 모델 중 무엇을 올릴 것인가?
---

# 최신 객체 탐지 모델 조사 (GitHub 기준, 2026-10-02)

## 결론 (한 문장)
지금 도구(ultralytics)에서 바로 돌아가는 **YOLO26(n, s)** 을 기준선(YOLOv8n, YOLO11n)과 함께 비교 실험에 올리고, 비YOLO 후보는 **RF-DETR(Apache-2.0) 하나만** 따로 시험해 보자는 것이 이 조사의 제안이다. 정확도는 우리 데이터로 아직 재지 않았으므로 "제안"이다.

## 배경
- Day 2 과제는 모델 여러 개(v5, v8, v11 등)를 같은 지표(mAP50, P, R, F1, 속도)로 비교해서 하나를 고르고 이유를 쓰는 것이다. 이 노트는 그 후보에 "더 새로운 모델이 있는가"를 GitHub에서 찾아본 것이다.
- 우리 조건: 추론은 PC(RTX 4070 Laptop, VRAM 8 GB)에서 하고, 입력은 로봇 카메라의 1280x720 압축 영상(평균 약 21 Hz)이다. 데이터는 RC카 사진 수백 장 규모가 될 것이다.

## 근거

숫자와 사실은 모두 이 표에 모았다. 모든 "인용" 값은 각 프로젝트가 **자기 조건에서** 낸 값이라 서로 직접 비교할 수 없다 (한계 절 참고).

### A. GitHub에서 확인한 상태 (2026-10-02 조회)

| 주장 | 값 / 내용 | 출처 | 확인일 | 구분 |
|---|---|---|---|---|
| ultralytics 저장소 | AGPL-3.0, ★62,156, 최신 릴리스 v8.4.171 (2026-10-01) | https://github.com/ultralytics/ultralytics (GitHub API) | 2026-10-02 | 인용 |
| YOLO26 가중치 공개 시점 | `yolo26n.pt`가 assets 릴리스 v8.4.0 (2026-01-13)에 있음. 릴리스 노트에서 YOLO26이 처음 언급된 것은 v8.3.204 (2025-09-30) | https://github.com/ultralytics/assets/releases , https://github.com/ultralytics/ultralytics/releases (GitHub API) | 2026-10-02 | 인용 |
| YOLO26 지원 작업 | 탐지, 인스턴스 분할, 시맨틱 분할, 단안 깊이, 분류, 자세, OBB | https://docs.ultralytics.com/models/yolo26/ | 2026-10-02 | 인용 |
| YOLO26 변경점 | DFL 제거, NMS 없는 추론(선택적 end-to-end), MuSGD 옵티마이저, ProgLoss와 STAL, 가벼운 헤드 | 같은 페이지 | 2026-10-02 | 인용 |
| YOLO26 COCO 성능 (mAP50-95 / T4 TensorRT10 ms / 파라미터 M) | n 40.9 / 1.7 / 2.4, s 48.6 / 2.5 / 9.5, m 53.1 / 4.7 / 20.4, l 55.0 / 6.2 / 24.8, x 57.5 / 11.8 / 55.7 (640 px) | 같은 페이지 | 2026-10-02 | 인용 |
| YOLO26 라이선스 | AGPL-3.0 및 Enterprise | 같은 페이지 | 2026-10-02 | 인용 |
| YOLO12 위치 | 커뮤니티 모델. Ultralytics는 "대부분의 production에는 YOLO11 또는 YOLO26"을 권장. 단점으로 학습 불안정, 메모리 많이 씀, CPU 느림을 적음 | https://docs.ultralytics.com/models/yolo12/ | 2026-10-02 | 인용 |
| YOLO12 COCO 성능 (mAP50-95 / T4 ms) | n 40.6 / 1.64, s 48.0 / 2.61, m 52.5 / 4.86, l 53.7 / 6.77, x 55.2 / 11.79 | 같은 페이지 | 2026-10-02 | 인용 |
| RF-DETR 저장소 | Apache-2.0 (패키지, N~L 가중치). XL, 2XL과 `rfdetr_plus`는 PML 1.0. 최신 1.11.1 (2026-09-30). ICLR 2026, arXiv 2511.09554 | https://github.com/roboflow/rf-detr (README, GitHub API) | 2026-10-02 | 인용 |
| RF-DETR COCO 성능 (AP50:95 / 지연 ms / 해상도) | N 48.4 / 2.3 / 384, S 53.0 / 3.5 / 512, M 54.7 / 4.4 / 576, L 56.5 / 6.8 / 704. 파라미터 N 30.5 M | 같은 README 표 (Roboflow 자체 측정) | 2026-10-02 | 인용 |
| RF-DETR 지연 측정 조건 | T4, TensorRT, FP16, 배치 1로 알려져 있으나 현재 README 문구에서는 직접 확인하지 못함 | 검색 결과 요약 | 2026-10-02 | 미확인 |
| RF100-VL (여러 분야 소규모 데이터 전이) AP50:95 | RF-DETR-N 57.7, YOLO11-N 55.3 | RF-DETR README 표 (Roboflow 자체 측정) | 2026-10-02 | 인용 |
| RF-DETR 학습 데이터 형식 | COCO JSON 또는 YOLO(`data.yaml`, `train/images`, `train/labels`, `valid/...`). 폴더 구조로 자동 판별. 학습 예: `RFDETRMedium().train(dataset_dir=..., epochs=..., batch_size="auto")`. 내보내기: ONNX, TensorRT, TFLite, OpenVINO 등 | https://rfdetr.roboflow.com/latest/learn/train/ | 2026-10-02 | 인용 |
| D-FINE | Apache-2.0, 마지막 push 2026-08-19, GitHub 릴리스 없음. COCO AP / 지연 ms: N 42.8 / 2.12, S 48.5 / 3.49, M 52.3 / 5.62, L 54.0 / 8.07, X 55.8 / 12.89. Objects365로 사전학습한 체크포인트는 상업 사용이 확정된 것이 아니라고 README가 적음 | https://github.com/Peterande/D-FINE | 2026-10-02 | 인용 |
| DEIMv2 | 자체 "DEIMv2 License"(GitHub가 SPDX로 인식 못 함, 상업 문의는 이메일), 마지막 push 2026-08-24. COCO AP: Atto 23.8, Femto 31.0, Pico 38.5, N 43.0, S 50.9, M 53.0, L 56.0, X 57.8 (N의 지연 2.32 ms, 조건 미확인) | https://github.com/Intellindust-AI-Lab/DEIMv2 | 2026-10-02 | 인용 |
| RT-DETRv4 | Apache-2.0 (GitHub 표기), ECCV 2026, 마지막 push 2026-07-06. COCO AP / T4 지연: S 49.8 / 3.66 ms, M 53.7 / 5.91, L 55.4 / 8.07, X 57.0 / 12.90. 비전 파운데이션 모델 증류 방식 | https://github.com/RT-DETRs/RT-DETRv4 (arXiv 2510.25257) | 2026-10-02 | 인용 |
| EdgeCrafter | 자체 "EdgeCrafter License", TMLR 2026 (arXiv 2603.18739), 마지막 push 2026-08-24. ECDet-S/M/L/X, Intel Geti(2026-08-13)와 LightlyTrain(2026-07-23)에 통합. 성능표는 수집하지 않음 | https://github.com/Intellindust-AI-Lab/EdgeCrafter | 2026-10-02 | 인용 (성능은 미확인) |
| 갱신이 오래된 저장소 | YOLOv10 마지막 push 2025-03-14, YOLOv9 2024-08-09, mmdetection 2024-08-21, YOLOX 2025-06-08, YOLOv13 2025-11-18 | 각 저장소 (GitHub API) | 2026-10-02 | 인용 |

### B. 우리 환경에서 직접 확인한 것

| 주장 | 값 / 내용 | 출처 | 확인일 | 구분 |
|---|---|---|---|---|
| 설치된 ultralytics 8.4.170이 지원하는 모델 설정 | `yolo26*.yaml`(탐지, seg, obb, pose, cls, depth, sem, p2/p6), `yoloe-26-seg`, `yolo12*`, `rtdetr-*`, `yolov10*`, `yolov9*` 포함 | 직접 측정 (패키지의 `cfg/models` 폴더 목록) | 2026-10-02 | 측정 |
| 우리 카메라 스트림 속도 | 압축 1280x720에서 평균 약 19~21 Hz, 프레임당 약 0.15 MB, 약 3.9 MB/s | 직접 측정 (`ros2 topic hz`, `bw`) | 2026-10-02 | 측정 |
| 우리 도구로 YOLO26, YOLO12 학습이 되는가 | `train_compare`가 `yolo26n.pt`, `yolo12n.pt`를 장난감 데이터(30장 검증 규모)로 학습(2 epoch, imgsz 320, 배치 8), 검증, 비교표 생성까지 오류 없이 끝냈다 (status ok). 점수는 2 epoch이라 의미 없음 | 직접 측정 (`train_compare --models yolo26n.pt yolo12n.pt`) | 2026-10-02 | 측정 |
| `pip install rfdetr`의 영향 | numpy, torch, torchvision, opencv, pillow, scipy, ultralytics는 바뀌지 않음. 새로 들어오는 것: transformers 5.18.0, supervision 0.30.6, huggingface_hub 1.33.0 등 | 직접 측정 (`pip install --dry-run rfdetr`, 실제 설치는 안 함) | 2026-10-02 | 측정 |

**우리 PC의 추론 시간** (RTX 4070 Laptop, ultralytics 8.4.170, imgsz 640, 배치 1, ultralytics 기본 샘플 이미지 1장(1080x810), 워밍업 20회 후 100회의 **중앙값**, 전처리+추론+후처리 합계 ms). 데스크톱이 GPU를 같이 쓰고 있어(측정 전 사용률 33~39%) 평균 대신 중앙값을 썼다.

| 모델 | FP32 | FP16 (`quantize=16`) |
|---|---|---|
| YOLOv8n | 4.09 | 3.93 |
| YOLO11n | 4.91 | 5.00 |
| YOLO12n | 6.63 | 6.93 |
| YOLO26n | 5.44 | 5.59 |
| YOLOv8s | 4.94 | 4.23 |
| YOLO11s | 5.00 | 5.26 |
| YOLO12s | 6.88 | 6.97 |
| YOLO26s | 5.76 | 6.06 |
| RT-DETR-L (ultralytics 내장) | 20.98 | 13.28 |

### C. 추정

| 주장 | 값 / 내용 | 출처 | 확인일 | 구분 |
|---|---|---|---|---|
| 속도는 n, s 모델 사이의 선택 기준이 아니다 | 위 모델은 모두 합계 4~7 ms이고, 카메라 프레임 간격은 약 47 ms (1 ÷ 21 Hz)이다. FP16은 이득이 없다 (배치 1에서는 연산량보다 호출 오버헤드가 크기 때문으로 보이나 확인하지 않음) | 위 측정과 21 Hz에서 계산 | 2026-10-02 | 추정 |
| YOLO12는 우선순위가 낮다 | 같은 크기에서 우리 GPU 시간이 가장 길고(n: 6.6 ms 대 4.1~5.4 ms), Ultralytics가 production을 권하지 않으며, 학습 불안정과 메모리 사용을 경고한다 | 위 인용과 측정 | 2026-10-02 | 추정 |

## 비교

| 후보 | 장점 | 단점 | 우리 상황에 맞나 |
|---|---|---|---|
| **YOLO26 (n, s)** | 지금 도구 그대로 쓴다(설정이 이미 들어 있고, `train_compare`로 학습, 검증, 비교표까지 돌아가는 것을 장난감 데이터로 확인했다. `yolo_infer`는 ultralytics 기반). NMS 없는 추론. 분할과 OBB도 같은 도구에 있다. 문서값 AP가 YOLO 계열 중 가장 높다 | AGPL-3.0. 우리 데이터에서의 성능은 미검증 | **1순위. 비교 실험에 올린다** |
| YOLO11n, YOLOv8n | 기준선. Ultralytics 권장(YOLO11), 강사 예제 기본값(v8) | 최신이 아님 | **기준선으로 같이 돌린다** |
| YOLO12 | AP는 YOLO11과 비슷 | 비권장, 우리 GPU에서 가장 느림, 학습 불안정 경고 | 시간이 남으면 1회만 |
| **RF-DETR (N, S)** | Apache-2.0. 소규모 데이터셋 전이 성능을 내세움(RF100-VL). YOLO 폴더 구조 데이터를 그대로 쓸 수 있다고 문서가 적음. ONNX, TensorRT 내보내기 | 별도 학습 코드와 추론 어댑터가 필요(우리 `yolo_infer`는 ultralytics 전용). 파라미터 30 M(N). 우리 데이터와 GPU에서 미검증 | **비YOLO 도전자 1개로 제안** |
| D-FINE, DEIMv2, RT-DETRv4, EdgeCrafter | DETR 계열 최신이고 표 값이 높다 | 각자 학습 파이프라인(우리 도구와 분리). DEIMv2와 EdgeCrafter는 자체 라이선스. ROS 통합 정보 없음 | 이번에는 관찰만 |
| RT-DETR-L (ultralytics 내장) | 같은 도구에서 실행 | 우리 GPU에서 21 ms로 가장 느림, 모델이 큼 | 제외 |

## 한계와 미확인

- 정확도와 지연 숫자는 각 프로젝트가 **자기 조건(T4, TensorRT, 해상도 등)** 에서 낸 값이라 서로, 그리고 우리 RTX 4070 측정과 직접 비교할 수 없다. **우리 RC카 데이터에서의 정확도는 아직 재지 않았다** (데이터 수집 중).
- 문서 페이지의 일부 표는 도구로 요약해서 읽었다. YOLO26의 핵심 값(n 40.9 AP, 1.7 ms, x 57.5 AP)은 다른 출처(논문 검색 결과 요약)와 일치하는 것을 확인했지만, 표 전체를 사람이 원문과 한 줄씩 대조하지는 않았다.
- YOLO26 문서 페이지는 출시일을 적지 않아서, 날짜는 GitHub 릴리스로 대신 확인했다 (2026-01-13에 가중치 공개). 공식 발표일은 확인하지 못했다.
- RF-DETR의 실제 설치, 학습, 우리 데이터 형식으로의 학습은 해 보지 않았다. dry-run은 의존성만 확인한 것이다.
- DEIMv2와 EdgeCrafter의 라이선스 전문을 읽지 않았다. EdgeCrafter와 RT-DETRv4의 상세 성능표는 수집하지 않았다.
- 속도 측정은 샘플 이미지 1장, 배치 1, GPU 공유 상태이다. 입력 크기가 640 고정이라 RF-DETR처럼 해상도가 모델마다 다른 경우와 같은 조건이 아니다.
- 2026-07 이후에 공개된 모델은 웹 검색에서 확인하지 못했다. GitHub에서 2026년에 만들어진 `object-detection` 주제 저장소를 별 수 순으로 훑었지만 새 아키텍처로 보이는 것은 EdgeCrafter 정도였고, 빠뜨린 후보가 있을 수 있다.

## 다음 단계

- [ ] `ROKEY_P4`의 비교 실험 yaml에 `yolo26n`, `yolo26s`를 추가한다.
- [x] 장난감 데이터로 `yolo26n`, `yolo12n` 학습이 우리 `train_compare`에서 도는지 스모크 테스트했다 (2026-10-02, 위 표). `yolo_infer`로 YOLO26 추론하는 것은 아직 확인하지 않았다.
- [ ] 실제 RC카 데이터(수집 중)로 `yolov8n`, `yolo11n`, `yolo26n`, `yolo26s`를 같은 지표로 비교한다.
- [ ] RF-DETR는 별도 가상환경에 설치해서 같은 데이터(YOLO 폴더 구조)로 N 모델을 1회 학습하고 mAP50와 추론 시간을 비교한다.
- [ ] 분할(seg)과 OBB는 탐지가 안정된 뒤에 검토한다 (YOLO26의 seg, obb 설정이 같은 도구에 있다).
- [ ] AGPL-3.0이 수업 프로젝트에서 문제가 되는지 팀에서 확인한다.

## 출처

1. Ultralytics YOLO26 문서, https://docs.ultralytics.com/models/yolo26/, 확인일 2026-10-02
2. Ultralytics YOLO12 문서, https://docs.ultralytics.com/models/yolo12/, 확인일 2026-10-02
3. ultralytics/ultralytics 및 ultralytics/assets 릴리스, https://github.com/ultralytics/ultralytics/releases , https://github.com/ultralytics/assets/releases, 확인일 2026-10-02
4. roboflow/rf-detr README와 문서, https://github.com/roboflow/rf-detr , https://rfdetr.roboflow.com/latest/learn/train/ , arXiv 2511.09554, 확인일 2026-10-02
5. Peterande/D-FINE README, https://github.com/Peterande/D-FINE, 확인일 2026-10-02
6. Intellindust-AI-Lab/DEIMv2 README, https://github.com/Intellindust-AI-Lab/DEIMv2, 확인일 2026-10-02
7. RT-DETRs/RT-DETRv4 README, https://github.com/RT-DETRs/RT-DETRv4 (arXiv 2510.25257), 확인일 2026-10-02
8. Intellindust-AI-Lab/EdgeCrafter README, https://github.com/Intellindust-AI-Lab/EdgeCrafter (arXiv 2603.18739), 확인일 2026-10-02
9. 후보 탐색에만 쓴 웹 검색 결과: YOLO26 분석 논문 arXiv 2601.12882, Roboflow 블로그 "Best Object Detection Models 2026", JetBrains 블로그 (2026-07), 확인일 2026-10-02
