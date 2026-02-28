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

> 🌐 **多語言狀態：** `i18n/` 已存在並保留作為語言專用 README 檔案。連結中的在地化文件仍在規劃/進行中。

## ✨ 一覽

| 焦點 | 位置 |
|---|---|
| 核心工作流程 | `notebooks/` |
| 環境規格 | `notebooks/reconstruction/lensless.yaml` |
| 模組說明 | `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/` |
| 入口文件 | `i18n/README.*.md` |

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

*左側為個人使用原型，右側為機構使用原型*

## 概覽

Lazeal OptiX 是一個面向醫療相關診斷場景的無透鏡影像工作流程研究/原型專案。該儲存庫目前以 notebook 為核心，屬於實驗性質，目標是在資源受限環境中讓進階診斷方法更易取得。

核心概念包括：

- 無透鏡影像重建
- 光源定位
- 多張影像比對與對齊

該專案主要透過 `notebooks/` 目錄中的 Jupyter notebook 來維護，模組化內容存放於各自目錄。

### 儲存庫狀態快照

| 項目 | 目前狀態 |
|---|---|
| 專案成熟度 | 研究原型 |
| 主要執行模式 | Jupyter notebook 工作流程 |
| 主要實驗領域 | 重建、光源定位、多圖比對 |
| 根目錄套件化/CI | 目前尚未宣告 |
| 多語言文件 | `i18n/` 目錄骨架已存在 |

## 功能特性

1. **進階顯微鏡概念**：用於精細分析的進階光學與影像擷取模式。
2. **生化 / 診斷情境**：目標於健康指標偵測的實驗工作流程。
3. **居家友善方向**：設計上著重可近用性與實際部署。
4. **筆記本優先體驗**：以 notebook 作為主要執行路徑。
5. **無透鏡重建工具**：提供高解析度重建的計算式流程。
6. **光源定位工具**：進行光源定位與幾何校正的實驗。
7. **多圖比對**：基於 SIFT 的比對、串接與對齊工具。

## 專案結構

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

### 模組說明

- `camera/`：用於高解析度樣本擷取的相機腳本/資源。
- `light_source/`：用於光源控制與最佳化的腳本/資源。
- `reconstruction/`：用於計算重建的腳本/資源。
- `three_axis_cnc/`：用於三軸 CNC 定位與控制的腳本/資源。
- `notebooks/`：實驗與方法的主要技術工作區。

## 筆記本

`notebooks` 目錄內包含紀錄核心實驗方法的 Jupyter notebooks。這些 notebook 提供各領域的程式碼、視覺化與方法說明。

### `light_source_location`

包含與光源位置估計相關的 notebook。這些方法可支援光源幾何校正與重建準確度的提升。

### `multiple_match`

包含用於影像/圖樣比對與對齊的 notebook 與腳本，以支援穩健的註冊流程。

### `reconstruction`

包含從擷取影像進行重建的 notebook，包含前處理與實驗腳本。

## 先決條件

- 作業系統：目前建議使用 Linux/macOS 以配合 Conda 與 OpenCV 工作流程。
- Python：環境目標為 **Python 3.7**。
- Conda：用於重現文件中的 `lensless` 環境。
- Jupyter Notebook/Lab。
- `multiple_match.cpp` 的可選 C++ 工具鏈：
  - 支援 C++17 的 `g++`。
  - 含 contrib 模組的 OpenCV 4.x（`opencv2/xfeatures2d.hpp` / SIFT）。

## 安裝

### 1) 取得專案

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) 建立 notebook 環境

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) 啟動 Jupyter

```bash
jupyter notebook
```

## 使用方式

該專案主要以開啟 notebook 並依文件順序逐格執行來進行。

### 重建流程

- 開啟 `notebooks/reconstruction/dataset_prep.ipynb` 進行資料集準備。
- 開啟 `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` 進行重建／訓練實驗。

### 光源定位流程

- 開啟 `notebooks/light_source_location/` 下的 notebook。

### 多重比對流程

- 開啟 `notebooks/multiple_match/` 下的 notebook。
- 可選工具：`notebooks/multiple_match/multiple_match.cpp`。

## 設定

### Conda 環境

主要環境規格：

- `notebooks/reconstruction/lensless.yaml`

主要相依套件包含：

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- notebook 中與 `opencv` 相關的電腦視覺工作流程相依套件

### 資料與路徑

- **假設：** 資料集存在於本機，且未在儲存庫根目錄集中宣告。
- **假設：** C++ 比對工具預期其執行路徑下有一個 `all/` 目錄，且其中是可讀取的灰階影像。

若你的本機設定不同，請更新 notebook 的路徑 cell 與 C++ 輸入目錄。

## 範例

### 執行比對工具

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

預期行為：

- 從 `all/` 讀取影像
- 計算跨影像的連鎖 SIFT 比對
- 輸出形如 `result_<timestamp>.png` 的結果影像

### 啟動指定 notebook

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## 開發說明

- 根目錄目前未提供套件化清單（`pyproject.toml`、`requirements.txt`、`setup.py`）或 CI/測試架構。
- 作業以實驗優先為主；notebook 是目前演算法的真實來源。
- `camera/`、`light_source/`、`reconstruction/` 與 `three_axis_cnc/` 包含元件級別說明，為日後擴充運行手冊提供良好切入點。
- `i18n/` 已預留給語言專屬文件。

## 疑難排解

- **Conda 解決相依問題：** 更新 Conda、確認 channel 順序，並重試環境建立。
- **Notebook 核心不一致：** 確認 Jupyter 使用的是 `lensless` 環境。
- **OpenCV/SIFT 編譯錯誤：** 安裝 OpenCV contrib 模組並確認 `opencv2/xfeatures2d.hpp` 可用。
- **Notebook 檔案找不到：** 驗證預期資料集與 notebook 相對路徑。
- **比對工具未讀到影像：** 確認 `notebooks/multiple_match/all/` 存在且包含有效影像檔。

## 路線圖

- 在 `camera/`、`light_source/`、`reconstruction/` 與 `three_axis_cnc/` 中擴展模組級運行手冊。
- 文件化資料集契約並提供可重現之範例資料參考。
- 為主要 notebook 流程補上腳本封裝。
- 為重建與比對結果加入驗證檢查。
- 完成 `i18n/` 下的多語言 README 文件。

## 參與貢獻

歡迎共同參與與貢獻。

- 開啟 issue 進行討論。
- 對範圍內的文件或實驗變更提出 pull request。
- 在進行大型硬體或協定層面的變更前，先與維護者聯繫。

## 貢獻指南

1. Fork 該儲存庫。
2. 建立一個 feature 分支。
3. 讓變更維持在明確範圍，並完成文件紀錄（尤其是 notebook）。
4. 提交 pull request，並說明動機、方法與任何驗證紀錄。

## 授權條款

目前儲存庫根目錄中尚未包含授權檔案。

**假設／待補：** 請新增 `LICENSE` 檔案，並以精確的 SPDX 識別碼更新本節。

## 聯絡方式

若有更多問題或合作意願，請寄信至 `contact@lazealoptix.com`。 


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
