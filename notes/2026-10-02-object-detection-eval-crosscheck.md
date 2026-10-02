---
title: 객체 검출 학습·평가 정리의 교차 검증 (수업 중 AI 대화 노트 검토)
date: 2026-10-02
status: draft   # draft | reviewed | final
question: 수업 중 AI 챗봇과 나눈 객체 검출 학습·평가 정리 노트의 주장 중 무엇이 확인되고, 무엇이 우리 프로젝트(TurtleBot4 카메라 + PC GPU 추론)에 맞지 않는가?
---

# 객체 검출 학습·평가 정리의 교차 검증 (수업 중 AI 대화 노트 검토)

> **검토 대상:** 수업 시간에 수강생이 AI 챗봇(제미나이)과 나눈 대화를 정리한 노트다. 내용은 에폭과 배치, 평가 지표, 학습 그래프 해석, TensorRT 배포, 최신 검출 모델 비교다.
> 원문은 이 저장소에 옮기지 않았고 주장만 요약했다. 다른 사람이 다시 확인하기 쉽도록 표에 **판정** 열을 뒀다. 판정은 `맞음` / `부분 확인` / `논쟁 중` / `미확인` / `우리 환경에 해당 없음` / `우리 환경에서 위험` 중 하나다.
> **한계:** 공식 문서와 논문 초록은 웹 가져오기 도구(요약기)로 읽었다. 표의 인용문이 원문과 글자까지 같은지는 사람이 한 번 더 대조해야 한다 (아래 "다음 단계").

## 결론 (한 문장)
에폭, 정밀도·재현율, mAP 정의 같은 개념 설명은 공식 문서와 맞지만, ① TensorRT와 Jetson 전제는 우리 로봇(Raspberry Pi 4)에 해당하지 않고, ② "mAP50 1.0 = 건전한 학습"이라는 해석은 우리 데이터가 작아(검증 16장, 테스트 8장) 점수가 포화되므로 위험하며(연속 프레임을 블록으로 묶어 나누는 분할 옵션은 이 노트 작성 뒤에 추가됐으나 세션 사이 일반화는 별도 테스트셋이 있어야 잰다), ③ 배치에 따라 학습률을 손으로 조정해야 한다는 설명은 Ultralytics에서는 일부 자동이라 고쳐야 한다.

## 근거

구분 값: `인용`(출처에서 가져옴), `측정`(직접 잼, 방법을 적는다), `추정`(계산식 또는 이유를 적는다), `미확인`

### A. 학습 메커니즘

| 주장 (노트 요약) | 확인한 내용 | 출처 | 확인일 | 구분 | 판정 |
|---|---|---|---|---|---|
| 에폭마다 검증해 가장 좋은 가중치를 저장하고, 개선이 없으면 조기 종료한다 | Ultralytics 학습 코드가 `last.pt`와 `best.pt` 경로를 두고(168행), `EarlyStopping(patience=...)`을 만들며(461행), `val` 옵션이 켜져 있거나 마지막 에폭이면 `validate()`를 호출해 `fitness`를 얻는다(649 ~ 660행). 설치본 기본값은 `patience: 100` | ultralytics `engine/trainer.py` (main 브랜치, 줄 번호는 바뀔 수 있음), 이 PC 설치본 8.4.170의 `cfg/default.yaml` | 2026-10-02 | 측정 (코드 읽기, 실행은 안 함) | 맞음 (Ultralytics 기준) |
| 큰 배치는 뾰족한 최솟값(sharp minima)으로 가서 일반화가 나쁘고, 작은 배치는 평평한 최솟값으로 간다 | 한 논문의 주장이다: "large-batch methods tend to converge to sharp minimizers ... sharp minima lead to poorer generalization". 반론 논문은 "most notions of flatness are problematic for deep models and can not be directly applied to explain generalization"이라고 쓴다 | arXiv 1609.04836 (Keskar 외), arXiv 1703.04933 (Dinh 외) | 2026-10-02 | 인용 | 논쟁 중 (정설이 아니라 가설로 써야 함) |
| 배치를 키우면 학습률도 비례해 키운다 (선형 스케일링 규칙), 또는 √배 | 초록: "adopt a hyper-parameter-free linear scaling rule for adjusting learning rates as a function of minibatch size". √배 대안은 이 초록에 없다 | arXiv 1706.02677 (Goyal 외) | 2026-10-02 | 인용 | 부분 확인 (선형 규칙만 확인, √배 미확인) |
| 배치를 바꾸면 학습률을 직접 조정해야 한다 | Ultralytics는 `nbs`(기본 64)를 기준으로 `accumulate = max(round(nbs / batch_size), 1)`을 계산해 기울기를 누적한 뒤 갱신하고(321행), weight decay를 `batch_size × accumulate / nbs`로 조정한다(322행). 예를 들어 `batch: 16`이면 4번 누적해 64 단위로 갱신한다. 이 구간에서 학습률(`lr0: 0.01`)을 배치로 스케일하는 코드는 보지 못했다 (파일 전체를 읽은 것은 아님) | 위 `trainer.py`, 설치본 `default.yaml`의 `nbs: 64`, `lr0: 0.01` | 2026-10-02 | 측정 (코드 읽기) | 우리 도구 기준으로는 수정 필요 (batch 64 이하에서는 누적이 자동) |

### B. 평가 지표

| 주장 (노트 요약) | 확인한 내용 | 출처 | 확인일 | 구분 | 판정 |
|---|---|---|---|---|---|
| 검출에서는 정확도(Accuracy)가 아니라 Precision, Recall, mAP를 쓴다 (배경 99% 예시) | 개념적 설명이며 출처가 붙어 있지 않다. 논리는 타당하나 이 표에서는 확인된 출처가 없다 | - | - | 미확인 (출처 없음) | 미확인 |
| Precision과 Recall의 정의 | 공식 문서: "Precision quantifies the proportion of true positives among all positive predictions", "Recall calculates the proportion of true positives among all actual positives" | docs.ultralytics.com `guides/yolo-performance-metrics` | 2026-10-02 | 인용 | 맞음 |
| mAP50-95는 IoU 0.5부터 0.95까지 10단계 평균이다 | 문서는 "average of the mean average precision calculated at varying IoU thresholds, ranging from 0.50 to 0.95"라고 쓴다. **단계 간격(0.05)과 10단계는 이 문서에 없고**, COCO 페이지 발췌에서도 확인하지 못했다 | 위 문서, cocodataset.org `#detection-eval` | 2026-10-02 | 인용 | 부분 확인 (범위만 확인) |
| F1-Confidence 곡선은 높고 넓을수록 좋다 | 문서는 곡선 해석 지침을 거의 주지 않는다 (PR 곡선은 정밀도와 재현율의 절충, F1 곡선은 오탐과 미탐의 균형을 보여 준다는 정도). "넓은 평탄 구간이 좋다"는 해석은 문서에 없다 | 위 문서 | 2026-10-02 | 인용 | 미확인 (일반론으로는 타당해 보이나 근거 없음) |
| 학습 그래프에서 mAP50이 1.0에 닿고 검증 손실이 반등하지 않았으므로 "과적합 없이 건전하다" | 이 결론은 근거가 부족하다. 작은 단일 클래스 데이터에서 1.0은 검증셋이 작거나 쉬울 때, 또는 학습과 검증에 거의 같은 사진이 섞였을 때도 나온다. **우리 쪽 수치:** 모델 3종(yolov8n, yolo11n, yolo26n)이 검증셋 16장에서 mAP50 0.993 ~ 0.995, 테스트셋 8장에서 정답 비율 1.000으로 모두 포화돼 모델을 구별하지 못한다 (분할: 학습 45 / 검증 16 / 테스트 8장, 테스트에는 box가 0개). **분할 도구의 변화:** 이 노트를 처음 쓴 15:38에는 `prepare_dataset`에 이미지 단위 무작위 분할(`random.Random(seed).shuffle`)만 있었고, 16:02 커밋에서 `--block-size` 옵션(연속 프레임을 블록 단위로 묶어 분할)이 추가됐다. `amr_v2_clean`은 블록 8로 나뉘었다. 기본값(0)은 여전히 이미지 단위 무작위다. 블록 분할은 같은 촬영 세션 안의 누수를 줄이지만 세션 사이 일반화(다른 날, 다른 조명, 움직이는 상황)는 못 잰다. 실제 그래프 이미지는 보지 못했고 수강생의 설명만 근거다 | 팀 도구 `prepare_dataset.py` 코드와 커밋 기록, 모델 비교표(`comparison.md`), `split_report.json` (비공개 저장소와 로컬 파일) | 2026-10-02 | 측정 (코드와 파일 읽기) + 추정 (점수 포화의 의미) | 부분 완화됨 (누수 방지 옵션은 반영, 점수 포화와 세션 간 일반화는 남음) |
| 클래스당 이미지 1500장 이상, 인스턴스 1만 개 이상이 권장된다 | 공식 문서(YOLOv5 튜토리얼 "Tips for Best Training Results"): "≥ 1500 images per class recommended", "≥ 10000 instances (labeled objects) per class recommended". 우리 학습셋 69장의 인스턴스는 box 24개, car 51개(분할 합)다. 이 권장은 YOLOv5 문서의 일반 지침이다 | docs.ultralytics.com `yolov5/tutorials/tips_for_best_training_results`, `split_report.json` | 2026-10-02 | 인용 + 측정 | 우리 데이터는 권장치보다 훨씬 작음 |
| 배경 사진은 약 0 ~ 10%가 오탐 감소에 도움이 된다 | 같은 문서: "We recommend about 0-10% background images to help reduce FPs". 같은 문서는 배포 환경을 대표해야 하고 모든 인스턴스를 라벨링해야 한다고 한다 | 같은 문서 | 2026-10-02 | 인용 | 맞음 |
| 현재 모델이 사람 다리(검은 바지)를 `box`로 오탐한다 | 주행 촬영 238장을 학습된 모델(yolo26n, `conf` 0.25 이상)로 다시 돌리면 `box`가 잡힌 사진은 4장뿐이고 모두 검은 바지가 화면 오른쪽에 보이는 구간이다 (신뢰도 0.43 ~ 0.73). 이 세션에는 실제 `box`가 없었다 | 로컬 추론 (사진은 저장소에 올리지 않음) | 2026-10-02 | 측정 | 우리 환경에서 위험 (오탐 사례, 배경 사진 필요성) |

### C. 배포와 하드웨어

| 주장 (노트 요약) | 확인한 내용 | 출처 | 확인일 | 구분 | 판정 |
|---|---|---|---|---|---|
| mAP가 가장 높은 모델이 현장에서 최선은 아니다 (속도, 지연, 자원 제약) | 일반론으로 타당하다. 우리 쪽 기준 수치: 로봇 카메라 압축 스트림이 PC에서 평균 21.3 Hz로 들어오므로 추론이 그보다 빠르면 추론이 병목이 아니다 | [Pi 프로파일 노트](2026-10-02-turtlebot4-pi-profile.md) | 2026-10-02 | 측정 (21.3 Hz, 조건당 1회) | 맞음 (우리 기준 수치 있음) |
| TensorRT `.engine`은 빌드한 장비와 런타임에 종속된다 | 공식 문서: "TensorRT engines are hardware- and runtime-specific", "do not treat an `.engine` file as a portable model format" | docs.ultralytics.com `integrations/tensorrt` | 2026-10-02 | 인용 | 맞음 |
| 변환 명령은 `format=engine half=True device=0` | 같은 문서는 `half`가 `quantize=16`으로 바뀐 것처럼 표기한다. 이 PC 설치본(8.4.170)의 `default.yaml`에는 `half:` 줄이 없고 `quantize:`가 있다. `half=True`가 호환 별칭으로 아직 동작하는지는 **실행해 보지 않았다** | 위 문서, 설치본 `default.yaml` | 2026-10-02 | 측정 (파일 읽기) | 부분 확인 (옵션 이름 변경, 동작 미확인) |
| TensorRT와 Jetson 같은 엣지 GPU로 로봇에 배포한다 | 우리 로봇의 Pi는 Raspberry Pi 4 Model B Rev 1.5이고 NVIDIA GPU가 없다(Pi 4에 NVIDIA GPU가 없다는 점은 하드웨어 상식에서 온 추정). 추론은 PC에서 하는 설계다 | [Pi 프로파일 노트](2026-10-02-turtlebot4-pi-profile.md) (모델 측정) | 2026-10-02 | 측정 + 추정 | 우리 환경에 해당 없음 (로봇에서는) |
| PC에서 `.engine`으로 가속한다 | 이 PC: NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB, torch 2.14.1+cu130에서 CUDA 사용 가능, ultralytics 8.4.170. 그러나 `rokey_venv`에서 `import tensorrt`가 `ModuleNotFoundError`로 실패했다. 지금은 `.engine`을 내보낼 수 없다 | `nvidia-smi`, 파이썬 임포트 | 2026-10-02 | 측정 | 가능하나 미설치. 필요성은 미확인 |
| 카메라 안의 VPU로 추론을 옮길 수 있다 (대안) | 카메라가 USB에서 `Intel Movidius MyriadX`(`03e7:2485`, 동작 중에는 `03e7:f63b`)로 보이는 것은 측정했다. 이 VPU에 우리 YOLO 모델을 올릴 수 있는지는 조사하지 않았다 | [Pi 프로파일 노트](2026-10-02-turtlebot4-pi-profile.md) | 2026-10-02 | 측정 (장치) | 미확인 (가능 여부) |

### D. 최신 모델 비교표

| 주장 (노트 요약) | 확인한 내용 | 출처 | 확인일 | 구분 | 판정 |
|---|---|---|---|---|---|
| YOLOv10은 NMS를 없앴다 (이중 라벨 할당) | 제목은 "YOLOv10: Real-Time End-to-End Object Detection". 초록: "we first present the consistent dual assignments for NMS-free training of YOLOs, which brings competitive performance and low inference latency simultaneously". 소속(칭화대)은 이 페이지에 나오지 않는다 | arXiv 2405.14458 | 2026-10-02 | 인용 | 맞음 (소속 미확인) |
| RT-DETRv2는 쿼리 기반이라 NMS가 필요 없다 | 제목은 "RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer". 초록은 NMS를 **언급하지 않는다**. 대신 디코더의 선택적 다중 스케일 특징 추출, 동적 데이터 증강, 스케일 적응 하이퍼파라미터를 제안한다고 쓴다. 소속(바이두)도 이 페이지에 없다 | arXiv 2407.17140 | 2026-10-02 | 인용 | 미확인 (NMS 불필요 여부, 소속) |
| YOLO-World는 텍스트로 지정하는 제로샷 검출이다 | 제목은 "YOLO-World: Real-Time Open-Vocabulary Object Detection". 초록: "excels in detecting a wide range of objects in a zero-shot manner with high efficiency", LVIS에서 35.4 AP, V100에서 52.0 FPS. 소속(텐센트)은 이 페이지에 없다 | arXiv 2401.17270 | 2026-10-02 | 인용 | 맞음 (소속 미확인) |
| YOLO11은 C3k2 백본과 C2PSA 블록을 쓴다 | 공식 YOLO11 문서 페이지에는 두 이름이 **나오지 않고**, "개선된 backbone과 neck" 정도로만 쓴다 | docs.ultralytics.com `models/yolo11` | 2026-10-02 | 인용 | 미확인 |
| YOLO11은 NMS가 필요하다 | 같은 페이지가 명시하지 않는다. 설치본 `default.yaml`의 `nms` 옵션 주석은 "None: external NMS (default); True: embed NMS on export; False: NMS-free head if avail"이다 | 위 페이지, 설치본 `default.yaml` | 2026-10-02 | 인용 + 측정 | 부분 확인 (기본이 외부 NMS라는 점만) |
| 비교표가 최신이다 | 같은 YOLO11 문서 페이지가 **YOLO26**을 "end-to-end NMS-free inference"의 최신 Ultralytics 모델로 언급한다. 노트의 비교표에는 없다 | 위 페이지 | 2026-10-02 | 인용 | 최신이 아님 |
| "TensorRT 적합성: 최상 / 우수 / 양호 / 보통" 등급 | 측정이나 출처가 붙어 있지 않은 의견이다 | - | - | 미확인 | 근거 없음 |

## 우리 프로젝트에 미치는 영향

| 사실 | 조치 후보 |
|---|---|
| 분할 도구에 `--block-size`가 추가됐고 `amr_v2_clean`은 블록 8로 나뉘었다 (기본값은 여전히 이미지 단위 무작위) | 새 데이터셋을 만들 때 `--block-size`를 켠다. 그리고 다른 시간, 움직이는 상황에서 찍은 **별도 세션 테스트셋**을 정해 학습에서 제외한다 (주행 촬영 세션이 후보) |
| 사진 크기를 1280 × 720으로 정했는데 기본 `imgsz`는 640이다 | `imgsz` 640과 1280을 비교 항목에 넣는다. 이 PC의 GPU 메모리는 8188 MiB라 큰 `imgsz`에서는 `batch`를 줄여야 할 수 있다 (실제 사용량은 안 쟀다) |
| `batch`를 바꿔도 `nbs: 64` 때문에 누적이 자동이다 | 비교 실험에서 `batch`는 주로 메모리와 속도를 바꾼다고 가정하되, 64를 넘기면 별개이므로 실험 기록에 `batch`와 `accumulate`를 같이 적는다 |
| 카메라 스트림이 21.3 Hz다 | 모델 선택 기준에 "추론 속도가 약 21 Hz 이상"을 넣는다 (더 빠르면 추론은 병목이 아님) |
| 이 PC에 `tensorrt`가 없다 | `.engine` 내보내기는 추론이 병목으로 드러난 뒤에 검토한다 |

## 한계와 미확인

- AI 챗봇 원문은 저장소에 없다. 슬라이드 그래프의 내용은 수강생의 설명에 의존했고 실제 그림은 보지 못했다.
- 공식 문서와 논문 초록은 요약기를 거쳐 읽었다. 인용문과 "문서에 없다"는 판정(C3k2, C2PSA, NMS 언급, 단계 간격 등)은 원문에서 다시 찾아 확인해야 한다.
- 팀 분할 도구(`prepare_dataset`)는 이 노트를 쓴 뒤 16:02에 바뀌었고, 해당 행은 18시대에 보정했다. 다른 도구도 바뀔 수 있으니 날짜와 커밋을 같이 확인한다.
- 소스 코드는 `main` 브랜치 기준이고, 이 PC 설치본은 8.4.170이다. 두 곳의 기본값(`nbs`, `patience`, `batch`, `imgsz`, `lr0`)이 같은 것은 확인했으나 코드는 **읽기만 했고 학습을 돌려 보지 않았다**.
- 모델 비교표의 속도·정확도 수치는 가져오지 않았다. 필요하면 각 저장소의 공식 표에서 확인일과 함께 가져와야 한다.
- 수업에서 요구한 조사 항목 중 "탐지 대신 분할(segmentation), 자세(pose), 회전 박스(obb)를 쓸 수 없는가"는 이 노트에 없다.

## 다음 단계

- [ ] **교차 검증:** 사람이 "미확인" 항목을 원문(논문 PDF, 공식 문서)에서 확인한다. 우선순위: mAP50-95의 단계 간격, YOLO11의 C3k2 / C2PSA, RT-DETRv2의 NMS 언급, 세 논문의 소속, `half`와 `quantize`의 동작
- [ ] 인용문 3 ~ 4개를 무작위로 골라 원문과 글자까지 같은지 대조한다
- [x] 연속 프레임 누수 방지: 분할 도구에 `--block-size`가 추가된 것을 확인했다 (`amr_v2_clean`은 블록 8)
- [ ] 별도 세션 테스트셋을 정해 학습에서 제외하고 라벨링한다 (주행 촬영 세션이 후보)
- [ ] 배경(네거티브) 사진과 사람 다리 오탐 대응을 정한다 (배경 사진 비율은 약 0 ~ 10%가 권장)
- [ ] `imgsz` 640과 1280 비교 실험의 계획과 GPU 메모리 한계를 확인한다
- [ ] `tensorrt` 설치와 `.engine` 필요성을 판단한다 (추론이 병목일 때만)
- [ ] OAK-D 카메라 VPU에서 YOLO를 실행할 수 있는지 별도 노트로 조사한다
- [ ] "탐지 대신 분할, 자세, 회전 박스" 조사를 별도 노트로 한다

## 출처

1. Ultralytics Docs, YOLO Performance Metrics, https://docs.ultralytics.com/guides/yolo-performance-metrics/, 확인일 2026-10-02
2. Ultralytics Docs, YOLO11, https://docs.ultralytics.com/models/yolo11/, 확인일 2026-10-02
3. Ultralytics Docs, TensorRT integration, https://docs.ultralytics.com/integrations/tensorrt/, 확인일 2026-10-02
4. Ultralytics 소스, `ultralytics/engine/trainer.py`와 `ultralytics/cfg/default.yaml`, https://github.com/ultralytics/ultralytics (main 브랜치), 확인일 2026-10-02. 이 PC 설치본 8.4.170의 `cfg/default.yaml`도 읽음
5. Wang 외, YOLOv10: Real-Time End-to-End Object Detection, https://arxiv.org/abs/2405.14458, 확인일 2026-10-02 (저자 목록은 확인하지 않음)
6. Lv 외, RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer, https://arxiv.org/abs/2407.17140, 확인일 2026-10-02
7. Cheng 외, YOLO-World: Real-Time Open-Vocabulary Object Detection, https://arxiv.org/abs/2401.17270, 확인일 2026-10-02
8. Keskar 외, On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima, https://arxiv.org/abs/1609.04836, 확인일 2026-10-02
9. Dinh 외, Sharp Minima Can Generalize For Deep Nets, https://arxiv.org/abs/1703.04933, 확인일 2026-10-02
10. Goyal 외, Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, https://arxiv.org/abs/1706.02677, 확인일 2026-10-02
11. COCO Dataset, Detection Evaluation, https://cocodataset.org/#detection-eval, 확인일 2026-10-02 (가져온 페이지 발췌에는 평가 지표 설명이 없었다)
12. 로봇과 PC 측정값: [Pi 프로파일 노트](2026-10-02-turtlebot4-pi-profile.md), `nvidia-smi`와 파이썬 임포트 확인 (이 PC, 2026-10-02)
