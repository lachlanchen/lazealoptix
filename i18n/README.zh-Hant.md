[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


> 🌐 **多語言狀態：** `i18n/` 已存在，並保留給各語言 README 使用。連結中的在地化文件目前為規劃中/進行中。

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

*個人使用原型（左）與機構使用原型（右）*

## 概覽

Lazeal OptiX 是一項創新的醫療科技專案。專案核心是開發一款裝置，讓使用者能在家中獲得進階診斷能力。透過先進顯微與生化分析技術，此裝置旨在協助及早偵測多種健康問題，進而改善整體醫療成效。

Lazeal OptiX 專案源自於降低病痛、並讓健康診斷更普及可及的承諾。透過讓個人具備主動管理健康所需的工具，我們希望共同促成更健康的社會。

此儲存庫目前以研究/原型導向為主，且工作流程以 notebook 為核心。多數實作細節與實驗紀錄皆位於 `notebooks/` 下的 Jupyter notebooks。

### 快速一覽

| 領域 | 目前狀態 |
|---|---|
| 專案成熟度 | 研究原型 |
| 主要執行模型 | Jupyter notebook 工作流程 |
| 主要實驗領域 | 重建、光源定位、多影像匹配 |
| 根目錄封裝/CI | 目前尚未宣告 |
| 多語系文件 | 已有 `i18n/` 目錄骨架 |

## 功能特色

1. **先進顯微：** 採用先進顯微技術進行精細分析。
2. **生化分析：** 深入的生化分析可偵測多種健康指標。
3. **使用者友善：** 為居家使用而設計，提供簡單易用的介面。
4. **小型且平價：** Lazeal OptiX 體積精巧且價格可負擔，讓進階診斷走入日常。
5. **無透鏡重建工作流程：** 以 notebook 為基礎的計算成像與重建流程。
6. **光源定位實驗：** 用於估測光源位置的最佳化 notebooks。
7. **多影像匹配工具：** 基於 notebook 與 C++ OpenCV 的特徵匹配/對齊流程。

## 儲存庫結構

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

### 模組說明

- `camera/`：與相機使用相關的腳本/資源，用於高解析樣本擷取。
- `light_source/`：與光源控制及最佳化相關的腳本/資源。
- `reconstruction/`：與計算重建相關的腳本/資源。
- `three_axis_cnc/`：與三軸 CNC 定位/控制相關的腳本/資源。
- `notebooks/`：實驗與方法的主要技術工作區。

## Notebooks

`notebooks` 目錄包含記錄 Lazeal OptiX 專案各面向的 Jupyter notebooks。這些 notebooks 包含程式碼、視覺化與方法細節說明，提供互動式方式來探索並理解專案。

### `light_source_location`

`light_source_location` 目錄包含與光源位置估測相關的 notebooks。這些 notebooks 展示用於精準估算光源位置的演算法與方法，而這是 Lazeal OptiX 專案中的關鍵環節。

### `multiple_match`

`multiple_match` 目錄包含與多張影像或圖樣匹配相關的 notebooks 與腳本。此部分涉及複雜演算法，以精準完成影像匹配與對齊，這對於由無透鏡成像系統重建高解析影像而言是必要步驟。

### `reconstruction`

`reconstruction` 目錄包含與 Lazeal OptiX 裝置所擷取影像重建相關的 notebooks。這些 notebooks 記錄了由無透鏡成像系統重建高解析影像所使用的進階計算技術。

## 先決條件

- 作業系統：目前 notebook 與 OpenCV 工作流程建議使用 Linux/macOS。
- Python：提供的環境檔目標版本為 **Python 3.7**。
- Conda：重現文件中的 `lensless` 環境所需。
- Jupyter Notebook/Lab。
- `multiple_match.cpp` 的可選 C++ 工具鏈：
  - 支援 C++17 的 `g++`。
  - 含 contrib 模組的 OpenCV 4.x（`opencv2/xfeatures2d.hpp` / SIFT）。

## 安裝

### 1) Clone

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) 建立 notebook 環境（建議）

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) 啟動 Jupyter

```bash
jupyter notebook
```

## 使用方式

此儲存庫主要透過開啟 notebooks，並依序執行各 cell 來使用。

### 重建路線

- 開啟 `notebooks/reconstruction/dataset_prep.ipynb` 進行資料集準備。
- 開啟 `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` 進行重建/訓練實驗。

### 光源定位路線

- 開啟 `notebooks/light_source_location/` 下的 notebooks。

### 多重匹配路線

- 開啟 `notebooks/multiple_match/` 下的 notebooks。
- 可選 C++ 工具：`notebooks/multiple_match/multiple_match.cpp`。

## 設定

### Conda 環境

主要環境規格檔位於：

- `notebooks/reconstruction/lensless.yaml`

此檔案中的關鍵相依訊號包含：

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- notebooks 中與 `opencv` 相關的電腦視覺工作流程相依套件

### 資料與路徑

- **Assumption:** notebooks 預期本機存在資料集/檔案，但未在儲存庫根目錄集中宣告。
- **Assumption:** C++ 匹配工具預期其執行路徑相對位置有 `all/` 目錄，且其中包含可讀取為灰階的影像。

若你的本機環境不同，請相應更新 notebook 中的路徑 cell 與 C++ 輸入目錄設定。

## 範例

### 執行匹配工具（範例）

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

預期行為：

- 從 `all/` 讀取影像
- 計算跨影像串接的 SIFT 匹配
- 輸出名為 `result_<timestamp>.png` 的結果影像

### 啟動特定 notebook

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## 開發備註

- 此儲存庫目前在根目錄沒有封裝設定（`pyproject.toml`、`requirements.txt` 或 `setup.py`），也沒有根目錄 CI/測試框架。
- 工作模式以實驗優先：多數演算法以 notebooks 為主要真實來源。
- `camera/`、`light_source/`、`reconstruction/` 與 `three_axis_cnc/` 目前提供高層級模組說明，後續可逐步擴充操作手冊。
- `i18n/` 已存在，並保留給多語系 README 版本。

## 疑難排解

- **Conda solve 問題：** 更新 Conda 後重試建立環境。
- **notebook 核心不匹配：** 確認需要時目前啟用的 kernel 為 `lensless`。
- **OpenCV/SIFT 編譯錯誤：** 安裝 OpenCV contrib 模組，並確認 `opencv2/xfeatures2d.hpp` 可用。
- **notebook 找不到檔案：** 檢查資料集路徑與 notebook cell 所需的相對目錄。
- **C++ matcher 讀不到影像：** 確認 `notebooks/multiple_match/all/` 存在且包含有效影像檔。

## 路線圖

- 擴充 `camera/`、`light_source/`、`reconstruction/` 與 `three_axis_cnc/` 的模組級操作手冊。
- 文件化資料集契約，並提供可重現的範例資料指引。
- 為關鍵 notebook 流程補上可重現腳本。
- 為重建與匹配輸出新增測試/驗證檢查。
- 完成 `i18n/` 下的多語系 README 檔案。

## 參與方式

我們歡迎協作與貢獻。如果你有興趣參與 Lazeal OptiX 專案，歡迎提交 issue 或 pull request，或直接與我們聯繫。

## 貢獻指南

1. Fork 此儲存庫。
2. 建立功能分支。
3. 將變更維持在明確範圍並附上文件（尤其是 notebooks 相關變更）。
4. 建立 pull request，說明動機、方法與驗證方式。

若你計畫進行重大硬體/協定變更，建議先開 issue 對齊方向。

## 支援

目前此儲存庫尚未宣告專屬的資助/贊助 metadata。

若日後有更新，應在此新增贊助與捐款資訊，同時不移除既有技術文件。

## 授權

目前儲存庫根目錄尚未包含授權檔案。

**Assumption/Action needed:** 請新增 `LICENSE` 檔，並以確切 SPDX identifier 更新本節內容。

## 聯絡方式

若有進一步詢問或合作意向，請透過 `contact@lazealoptix.com` 與我們聯繫。
