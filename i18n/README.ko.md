[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Lazeal OptiX

![Project Stage](https://img.shields.io/badge/stage-research%20prototype-orange)
![Primary Workflow](https://img.shields.io/badge/workflow-notebook--centric-blue)
![Python](https://img.shields.io/badge/python-3.7-blue)
![Conda](https://img.shields.io/badge/environment-conda-44A833)
![Jupyter](https://img.shields.io/badge/interface-jupyter-F37626)
![OpenCV](https://img.shields.io/badge/cv-opencv%204.x-5C3EE8)
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![Localization](https://img.shields.io/badge/localization-11%20languages-8A4FFF)
![Platform](https://img.shields.io/badge/platform-linux%2FmacOS-2D9CDB)

> 🌐 **다국어 상태:** `i18n/`가 존재하며 언어별 README 저장을 위해 준비되어 있습니다. 다국어 문서는 계획/작업 중입니다.

## ✨ 한눈에 보기

| 항목 | 위치 |
|---|---|
| 핵심 워크플로우 | `notebooks/` |
| 환경 사양 | `notebooks/reconstruction/lensless.yaml` |
| 구성 요소 노트 | `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/` |
| 진입 문서 | `i18n/README.*.md` |

<table width="100%">
  <tr>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_individual.jpg" alt="Prototype for Individuals" style="width: 90%" />
    </td>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_institute.png" alt="Prototype for Institutions" style="width: 90%" />
    </td>
  </tr>
</table>

*좌측: 개인용 프로토타입, 우측: 기관용 프로토타입*

## 개요

Lazeal OptiX는 의료 진단에 인접한 분야에서 렌즈리스 이미징 워크플로우를 연구하는 연구/프로토타입 프로젝트입니다. 현재 저장소는 주로 노트북 기반이며 실험적 성격으로, 제한된 환경에서도 고급 진단 접근성을 높이는 데 목적이 있습니다.

핵심 아이디어는 다음과 같습니다.

- 렌즈리스 이미지 재구성
- 광원 위치 추정
- 다중 이미지 매칭 및 정렬

이 저장소는 주로 `notebooks/` 아래의 Jupyter 노트북으로 유지되며, 모듈별 맥락은 각 하위 디렉터리에 정리되어 있습니다.

### 저장소 상태 요약

| 항목 | 현재 상태 |
|---|---|
| 프로젝트 성숙도 | 연구 프로토타입 |
| 주요 실행 모델 | Jupyter 노트북 워크플로우 |
| 주요 실험 영역 | 재구성, 광원 위치 추정, 다중 이미지 매칭 |
| 루트 레벨 패키징/CI | 현재 미선언 |
| 다국어 문서 | `i18n/` 디렉터리 스캐폴드 존재 |

## 특징

1. **고급 현미경 개념**: 정밀 분석을 위한 고급 광학 및 촬영 패턴.
2. **생화학/진단적 맥락**: 건강 지표 탐지에 초점을 둔 실험 워크플로우.
3. **가정 중심 지향**: 접근성과 실용적 배치를 염두에 둔 설계.
4. **랩톱 우선 경험**: 노트북이 주 실행 경로입니다.
5. **렌즈리스 재구성 도구**: 고해상도 재구성을 위한 계산 파이프라인.
6. **광원 위치 추정 도구**: 광원 정밀 추정과 기하 보정 실험.
7. **다중 이미지 매칭**: SIFT 기반 매칭, 체이닝, 정렬 유틸리티.

## 프로젝트 구조

```text
lazealoptix/
├── README.md
├── prototype_individual.jpg
├── prototype_institute.png
├── figs/
│   ├── banner.svg|png
│   ├── logo.svg|png
│   └── logo-w-text.svg|png
├── camera/
│   └── README.md
├── light_source/
│   └── README.md
├── reconstruction/
│   └── README.md
├── three_axis_cnc/
│   └── README.md
├── notebooks/
│   ├── light_source_location/
│   │   ├── light_source_location_estimator_v1.4.ipynb
│   │   ├── light_source_location_estimator_varied_heights_v1.1.4.ipynb
│   │   └── light_source_location_estimator_varied_heights_v1.1.7.ipynb
│   ├── multiple_match/
│   │   ├── multiple_all_combination_v2.ipynb
│   │   ├── multiple_match.cpp
│   │   ├── multiple_match_centeralized_v1.6.ipynb
│   │   └── multiple_match_chain_v1.5.ipynb
│   └── reconstruction/
│       ├── dataset_prep.ipynb
│       ├── lensless.yaml
│       └── lensless-dropout-one-led-mahuichong.ipynb
└── i18n/
```

### 모듈 노트

- `camera/`: 고해상도 샘플 촬영을 위한 카메라 사용 관련 스크립트/리소스.
- `light_source/`: 광원 제어 및 최적화를 위한 스크립트/리소스.
- `reconstruction/`: 계산 기반 재구성 스크립트/리소스.
- `three_axis_cnc/`: 3축 CNC 위치 제어/운영 스크립트/리소스.
- `notebooks/`: 실험과 방법론의 주요 기술 작업 공간.

## 노트북

`notebooks` 디렉터리에는 핵심 실험 방법을 문서화한 Jupyter 노트북이 포함됩니다. 각 노트북은 코드, 시각화, 방법 노트를 제공해 해당 주제를 설명합니다.

### `light_source_location`

광원 위치 추정을 다루는 노트북을 포함합니다. 이 방법은 광원 기하 보정과 재구성 정확도 향상에 활용됩니다.

### `multiple_match`

이미지/패턴 매칭과 정렬을 위한 노트북과 스크립트를 제공합니다. 이를 통해 정합성이 강건한 등록 워크플로우를 지원합니다.

### `reconstruction`

촬영 이미지로부터의 재구성을 다루는 노트북을 포함하며, 전처리 및 실험 스크립트를 다룹니다.

## 사전 요구사항

- OS: 현재 Conda 및 OpenCV 워크플로우를 기준으로 Linux/macOS 권장.
- Python: 환경은 **Python 3.7**을 대상으로 합니다.
- Conda: 문서화된 `lensless` 환경을 재현하기 위해 필요합니다.
- Jupyter Notebook/Lab.
- `multiple_match.cpp`용 선택적 C++ 툴체인:
  - C++17을 지원하는 `g++`
  - OpenCV 4.x의 contrib 모듈 (`opencv2/xfeatures2d.hpp` / SIFT)

## 설치

### 1) 클론

```bash

git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) 노트북 환경 생성

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Jupyter 실행

```bash
jupyter notebook
```

## 사용법

이 저장소는 주로 노트북을 열고 문서화된 순서대로 셀을 실행하는 방식으로 사용합니다.

### 재구성 트랙

- `notebooks/reconstruction/dataset_prep.ipynb`를 열어 데이터셋을 준비합니다.
- `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb`를 열어 재구성/학습 실험을 수행합니다.

### 광원 위치 추적 트랙

- `notebooks/light_source_location/` 아래 노트북을 엽니다.

### 다중 매칭 트랙

- `notebooks/multiple_match/` 아래 노트북을 엽니다.
- 선택 유틸리티: `notebooks/multiple_match/multiple_match.cpp`

## 구성

### Conda 환경

주요 환경 명세:

- `notebooks/reconstruction/lensless.yaml`

주요 의존성 항목:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- 노트북에서 사용하는 OpenCV 기반 컴퓨터 비전 워크플로우 의존성

### 데이터와 경로

- **가정:** 데이터셋은 로컬에 있으며, 저장소 루트에서 중앙에서 선언되지 않습니다.
- **가정:** C++ 매칭 유틸리티는 실행 경로 기준 상대 경로의 `all/` 폴더에 회색조로 읽을 수 있는 이미지가 존재한다고 가정합니다.

로컬 환경이 다르면 노트북의 경로 셀과 C++ 입력 디렉터리를 해당 환경에 맞게 수정하세요.

## 예제

### 매칭 유틸리티 실행

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

예상 동작:

- `all/`에서 이미지 읽기
- 이미지 전반에 걸쳐 체인형 SIFT 매칭 계산
- `result_<timestamp>.png` 형식의 출력 이미지 생성

### 특정 노트북 실행

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## 개발 노트

- 루트 레벨 패키징 매니페스트(`pyproject.toml`, `requirements.txt`, `setup.py`) 또는 CI/테스트 하네스가 현재 없습니다.
- 작업은 실험 우선이며, 노트북이 현재 알고리즘의 사실상 소스 오브 트루스입니다.
- `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/`에는 모듈 수준의 설명이 있으며, 향후 실행 매뉴얼(런북)을 확장하기 좋은 진입점입니다.
- `i18n/`는 언어별 문서화를 위해 준비되어 있습니다.

## 문제 해결

- **Conda 의존성 해결 문제:** Conda를 업데이트하고 채널 순서를 확인한 뒤 환경 생성을 다시 시도하세요.
- **노트북 커널 불일치:** Jupyter가 `lensless` 환경을 사용 중인지 확인하세요.
- **OpenCV/SIFT 컴파일 오류:** OpenCV contrib 모듈을 설치하고 `opencv2/xfeatures2d.hpp` 사용 가능 여부를 확인하세요.
- **노트북 파일을 찾을 수 없음:** 기대 데이터셋과 노트북 상대 경로를 확인하세요.
- **매처가 이미지를 읽지 못함:** `notebooks/multiple_match/all/`에 유효한 이미지 파일이 있는지 확인하세요.

## 로드맵

- `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/`의 모듈별 런북을 확장합니다.
- 데이터셋 계약을 문서화하고 재현 가능한 샘플 데이터 참조를 제공합니다.
- 주요 노트북 파이프라인을 위한 스크립트 래퍼를 추가합니다.
- 재구성 및 매칭 결과의 유효성 검증 체크를 추가합니다.
- `i18n/` 아래 다국어 README 파일을 완료합니다.

## 참여 방법

협업과 기여를 환영합니다.

- 이슈를 열어 논의하세요.
- 문서 또는 실험 변경의 범위를 정해 풀 리퀘스트를 제출하세요.
- 하드웨어/프로토콜 수준 변경은 대형 리팩터링 전 유지보수자에게 먼저 연락하세요.

## 기여

1. 저장소를 포크하세요.
2. 기능 브랜치를 생성하세요.
3. 변경 범위를 유지하고 문서화하세요(특히 노트북 변경 시).
4. 동기, 방법, 검증 노트를 포함해 풀 리퀘스트를 제출하세요.

## 라이선스

현재 저장소 루트에는 라이선스 파일이 없습니다.

**가정/필요 조치:** `LICENSE` 파일을 추가하고 이 섹션을 정확한 SPDX 식별자로 업데이트하세요.

## 문의

추가 문의나 협업 제안은 `contact@lazealoptix.com`으로 보내주세요.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
