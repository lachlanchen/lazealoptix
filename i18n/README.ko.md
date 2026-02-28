[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


> 🌐 **다국어 상태:** `i18n/` 디렉터리는 언어별 README 파일을 위해 마련되어 있습니다. 링크된 현지화 문서는 계획/진행 중입니다.

<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# Lazeal OptiX

![Project Stage](https://img.shields.io/badge/stage-research%20prototype-orange)
![Primary Workflow](https://img.shields.io/badge/workflow-notebook--centric-blue)
![Python](https://img.shields.io/badge/python-3.7-blue)
![Conda](https://img.shields.io/badge/environment-conda-44A833)
![Jupyter](https://img.shields.io/badge/interface-jupyter-F37626)
![OpenCV](https://img.shields.io/badge/cv-opencv%204.x-5C3EE8)
![License](https://img.shields.io/badge/license-TBD-lightgrey)

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

*개인용 프로토타입(왼쪽)과 기관용 프로토타입(오른쪽)*

## 개요

Lazeal OptiX는 혁신적인 헬스케어 기술 프로젝트입니다. 이 프로젝트의 핵심은 사용자가 집에서 편안하게 고급 진단을 받을 수 있도록 돕는 장치 개발에 있습니다. 고급 현미경 기술과 생화학 분석 기법을 활용해 다양한 건강 문제를 조기에 발견할 수 있도록 하며, 이를 통해 의료 결과 개선에 기여하는 것을 목표로 합니다.

Lazeal OptiX 프로젝트는 고통을 줄이고 건강 진단의 접근성을 넓히겠다는 의지에서 시작되었습니다. 개인이 자신의 건강을 주도적으로 관리할 수 있는 도구를 제공함으로써, 더 건강한 사회를 만드는 데 기여하고자 합니다.

현재 이 저장소는 연구/프로토타입 중심이며 노트북 중심으로 운영됩니다. 대부분의 구현 세부사항과 실험은 `notebooks/` 아래의 Jupyter 노트북에서 관리됩니다.

### 한눈에 보기

| 영역 | 현재 상태 |
|---|---|
| 프로젝트 성숙도 | 연구 프로토타입 |
| 주요 실행 모델 | Jupyter 노트북 워크플로우 |
| 주요 실험 도메인 | 재구성, 광원 위치 추정, 다중 이미지 매칭 |
| 루트 패키징/CI | 현재 선언되지 않음 |
| 다국어 문서 | `i18n/` 디렉터리 스캐폴드 존재 |

## 기능

1. **고급 현미경 분석:** 고급 현미경 기법을 활용해 정밀한 분석을 수행합니다.
2. **생화학 분석:** 심층 생화학 분석으로 다양한 건강 지표를 감지할 수 있습니다.
3. **사용자 친화성:** 가정에서 사용할 수 있도록 간단하고 접근성 높은 사용자 경험을 제공합니다.
4. **소형·합리적 비용:** Lazeal OptiX는 소형이고 가격 접근성이 높아, 고급 진단 기술을 일상 사용자에게 제공합니다.
5. **렌즈리스 재구성 워크플로우:** 노트북 기반 계산 이미징 및 재구성 파이프라인을 제공합니다.
6. **광원 위치 추정 실험:** 광원 위치 추정을 위한 최적화 노트북을 포함합니다.
7. **다중 이미지 매칭 유틸리티:** 특징 매칭/정렬을 위한 노트북 및 C++ OpenCV 워크플로우를 제공합니다.

## 저장소 구조

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
│   ├── multiple_match/
│   └── reconstruction/
└── i18n/
```

### 모듈 메모

- `camera/`: 고해상도 샘플 캡처를 위한 카메라 사용 관련 스크립트/리소스.
- `light_source/`: 광원 제어 및 최적화를 위한 스크립트/리소스.
- `reconstruction/`: 계산 기반 재구성을 위한 스크립트/리소스.
- `three_axis_cnc/`: 3축 CNC 위치 제어/운영을 위한 스크립트/리소스.
- `notebooks/`: 실험과 방법론을 위한 핵심 기술 작업 공간.

## 노트북

`notebooks` 디렉터리에는 Lazeal OptiX 프로젝트의 다양한 측면을 설명하는 Jupyter 노트북이 포함되어 있습니다. 이 노트북들은 코드, 시각화, 그리고 프로젝트 방법론에 대한 상세 설명을 담고 있으며, 프로젝트를 인터랙티브하게 탐색하고 이해할 수 있는 수단을 제공합니다.

### `light_source_location`

`light_source_location` 디렉터리에는 광원 위치 추정과 관련된 노트북이 포함되어 있습니다. 이 노트북들에는 광원 위치를 정확히 추정하기 위한 알고리즘과 방법이 담겨 있으며, 이는 Lazeal OptiX 프로젝트의 핵심 요소입니다.

### `multiple_match`

`multiple_match` 디렉터리에는 여러 이미지 또는 패턴의 매칭과 관련된 노트북 및 스크립트가 포함되어 있습니다. 이 영역은 이미지들을 정확히 매칭하고 정렬하기 위한 복잡한 알고리즘을 다루며, 렌즈리스 이미징 시스템에서 고해상도 이미지를 재구성하기 위해 필요합니다.

### `reconstruction`

`reconstruction` 디렉터리에는 Lazeal OptiX 장치로 캡처한 이미지를 재구성하는 노트북이 포함되어 있습니다. 이 노트북들은 렌즈리스 이미징 시스템에서 고해상도 이미지를 복원하기 위해 사용되는 고급 계산 기법을 문서화합니다.

## 사전 요구사항

- OS: 현재 노트북 및 OpenCV 워크플로우 기준으로 Linux/macOS 권장.
- Python: 제공된 환경 파일은 **Python 3.7**을 대상으로 합니다.
- Conda: 문서화된 `lensless` 환경 재현에 필요합니다.
- Jupyter Notebook/Lab.
- `multiple_match.cpp`를 위한 선택적 C++ 툴체인:
  - C++17을 지원하는 `g++`.
  - contrib 모듈이 포함된 OpenCV 4.x (`opencv2/xfeatures2d.hpp` / SIFT).

## 설치

### 1) 클론

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) 노트북 환경 생성 (권장)

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Jupyter 시작

```bash
jupyter notebook
```

## 사용 방법

이 저장소는 주로 노트북을 열어 셀을 순서대로 실행하는 방식으로 사용합니다.

### 재구성 트랙

- 데이터셋 준비를 위해 `notebooks/reconstruction/dataset_prep.ipynb`를 엽니다.
- 재구성/학습 실험을 위해 `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb`를 엽니다.

### 광원 위치 추정 트랙

- `notebooks/light_source_location/` 아래 노트북을 엽니다.

### 다중 매칭 트랙

- `notebooks/multiple_match/` 아래 노트북을 엽니다.
- 선택적 C++ 유틸리티: `notebooks/multiple_match/multiple_match.cpp`.

## 구성

### Conda 환경

기본 환경 명세 파일 위치:

- `notebooks/reconstruction/lensless.yaml`

이 파일에서 확인할 수 있는 주요 의존성 신호:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- 노트북의 컴퓨터 비전 워크플로우와 연관된 `opencv` 계열 의존성

### 데이터 및 경로

- **가정:** 노트북은 저장소 루트에서 일괄 선언되지 않은 로컬 데이터셋/파일을 기대합니다.
- **가정:** C++ 매칭 유틸리티는 실행 경로 기준 상대 위치의 `all/` 디렉터리에 그레이스케일로 읽을 수 있는 이미지를 기대합니다.

로컬 설정이 다르다면 노트북 내 경로 셀과 C++ 입력 디렉터리를 해당 환경에 맞게 수정하세요.

## 예제

### 매칭 유틸리티 실행 (예시)

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

예상 동작:

- `all/`에서 이미지를 읽음
- 이미지 전반에 걸쳐 체인형 SIFT 기반 매칭 수행
- `result_<timestamp>.png` 형식의 출력 이미지 생성

### 특정 노트북 실행

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## 개발 노트

- 저장소 루트에는 현재 패키징 파일(`pyproject.toml`, `requirements.txt`, `setup.py`)과 CI/테스트 하네스가 없습니다.
- 작업 방식은 실험 우선이며, 대부분의 알고리즘에서 노트북이 사실상의 단일 진실 소스(source-of-truth)입니다.
- `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/`는 현재 상위 수준 모듈 설명을 제공하며, 점진적으로 운영 런북을 확장할 수 있습니다.
- `i18n/`은 다국어 README 변형을 위해 존재하며 해당 용도로 예약되어 있습니다.

## 문제 해결

- **Conda 해결(의존성) 문제:** Conda를 업데이트한 뒤 환경 생성을 다시 시도하세요.
- **노트북 커널 불일치:** 필요한 경우 활성 커널이 `lensless`와 일치하는지 확인하세요.
- **OpenCV/SIFT 컴파일 오류:** OpenCV contrib 모듈을 설치하고 `opencv2/xfeatures2d.hpp` 사용 가능 여부를 확인하세요.
- **노트북 파일 찾기 오류:** 노트북 셀에서 기대하는 데이터셋 경로와 상대 디렉터리를 점검하세요.
- **C++ 매처가 이미지를 읽지 못함:** `notebooks/multiple_match/all/` 경로 존재 여부와 유효 이미지 파일 포함 여부를 확인하세요.

## 로드맵

- `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/`의 모듈별 런북 확장.
- 데이터셋 계약(contract) 문서화 및 재현 가능한 샘플 데이터 포인터 제공.
- 핵심 노트북 파이프라인용 재현 가능한 스크립트 추가.
- 재구성 및 매칭 결과에 대한 테스트/검증 체크 추가.
- `i18n/` 아래 다국어 README 파일 완성.

## 참여하기

협업과 기여를 환영합니다. Lazeal OptiX 프로젝트에 참여하고 싶다면 이슈나 풀 리퀘스트를 제출하거나, 직접 연락해 주세요.

## 기여 가이드

1. 저장소를 포크합니다.
2. 기능 브랜치를 생성합니다.
3. 변경 범위를 명확히 유지하고 문서화합니다(특히 노트북 변경).
4. 동기, 방법, 검증을 설명하는 풀 리퀘스트를 엽니다.

하드웨어/프로토콜의 큰 변경을 계획한다면 정렬을 위해 먼저 이슈를 여는 것을 권장합니다.

## 지원

이 저장소에는 현재 전용 펀딩/스폰서십 메타데이터가 선언되어 있지 않습니다.

향후 변경될 경우, 기존 기술 문서를 제거하지 않은 채 이 섹션에 스폰서십 및 후원 정보를 추가해야 합니다.

## 라이선스

현재 저장소 루트에는 라이선스 파일이 없습니다.

**가정/필요 조치:** `LICENSE` 파일을 추가하고 이 섹션을 정확한 SPDX 식별자로 업데이트하세요.

## 문의

추가 문의 또는 협업 제안은 `contact@lazealoptix.com`으로 연락해 주세요.
