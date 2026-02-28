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

> 🌐 **多语言状态：** `i18n/` 已存在并保留用于语言特定的 README 文件。链接中的本地化文档仍处于规划/进行中。

## ✨ 概览

| 关注点 | 位置 |
|---|---|
| 核心工作流 | `notebooks/` |
| 环境规格 | `notebooks/reconstruction/lensless.yaml` |
| 组件说明 | `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/` |
| 入口文档 | `i18n/README.*.md` |

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

*左侧为个人使用原型，右侧为机构使用原型*

## 概览

Lazeal OptiX 是一个面向医疗相关诊断场景的无透镜成像工作流研究/原型项目。仓库目前以 notebook 为核心，属于实验性质，目标是在受限环境下让先进诊断方法更易于使用。

核心思路包括：

- 无透镜图像重建
- 光源定位
- 多图匹配与对齐

该仓库主要在 `notebooks/` 目录下以 Jupyter notebook 的形式维护，模块化上下文存放在各自目录中。

### 仓库状态快照

| 区域 | 当前状态 |
|---|---|
| 项目成熟度 | 研究原型 |
| 主要执行模型 | Jupyter notebook 工作流 |
| 主要实验领域 | 重建、光源定位、多图匹配 |
| 根目录打包/CI | 当前未声明 |
| 多语言文档 | `i18n/` 目录骨架已存在 |

## 功能特性

1. **先进显微概念**：用于精细分析的高级光学与成像采集模式。
2. **生化/诊断场景**：面向健康指标识别的实验流程。
3. **居家友好方向**：面向易用性与实际部署场景设计。
4. **笔记本优先体验**：以 notebook 作为主要执行路径。
5. **无透镜重建工具**：用于高分辨率重建的计算流程。
6. **光源定位工具**：进行光源定位与几何标定相关实验。
7. **多图匹配**：基于 SIFT 的匹配、链式匹配与对齐工具。

## 项目结构

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

### 模块说明

- `camera/`：用于高分辨率样本采集的相机脚本/资源。
- `light_source/`：用于光源控制与优化的脚本/资源。
- `reconstruction/`：用于计算重建的脚本/资源。
- `three_axis_cnc/`：用于三轴 CNC 定位与控制的脚本/资源。
- `notebooks/`：实验与方法的主要技术工作区。

## Notebooks

`notebooks` 目录包含记录核心实验方法的 Jupyter notebooks。这些 notebook 提供各领域的代码、可视化和方法说明。

### `light_source_location`

包含与光源位置估计相关的 notebooks。这些方法可支持光源几何标定和重建精度提升。

### `multiple_match`

包含用于图像/图案匹配与对齐的 notebooks 与脚本，支持稳健配准流程。

### `reconstruction`

包含从采集图像进行重建的 notebooks，包括预处理和实验脚本。

## 先决条件

- 操作系统：当前 Conda 与 OpenCV 工作流建议使用 Linux/macOS。
- Python：环境目标版本为 **Python 3.7**。
- Conda：用于复现文档中的 `lensless` 环境。
- Jupyter Notebook/Lab。
- `multiple_match.cpp` 的可选 C++ 工具链：
  - 支持 C++17 的 `g++`。
  - OpenCV 4.x（含 contrib 模块，`opencv2/xfeatures2d.hpp` / SIFT）。

## 安装

### 1) 克隆

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) 创建 notebook 环境

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) 启动 Jupyter

```bash
jupyter notebook
```

## 使用方式

本仓库主要通过打开 notebook 并按文档顺序逐个执行 cell 来使用。

### 重建路径

- 打开 `notebooks/reconstruction/dataset_prep.ipynb` 进行数据集准备。
- 打开 `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` 进行重建/训练实验。

### 光源定位路径

- 打开 `notebooks/light_source_location/` 下的 notebook。

### 多重匹配路径

- 打开 `notebooks/multiple_match/` 下的 notebook。
- 可选工具：`notebooks/multiple_match/multiple_match.cpp`。

## 配置

### Conda 环境

主要环境规格文件：

- `notebooks/reconstruction/lensless.yaml`

主要依赖包括：

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- notebook 中的 `opencv` 相关计算机视觉工作流依赖。

### 数据与路径

- **假设：** 数据集在本地存在，但仓库根目录未集中声明。
- **假设：** C++ 匹配工具期望其执行路径下存在 `all/` 目录，并且其中是可读的灰度图像。

如果你的本地环境不同，请按需更新 notebook 中的路径 cell 和 C++ 输入目录。

## 示例

### 运行匹配工具

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

预期行为：

- 从 `all/` 读取图像
- 计算跨图像链式 SIFT 匹配
- 生成类似 `result_<timestamp>.png` 的输出图像

### 启动指定 notebook

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## 开发说明

- 根目录目前未提供打包清单（`pyproject.toml`、`requirements.txt`、`setup.py`）或 CI/测试框架。
- 工作以实验优先为主；notebooks 是当前算法的真实来源。
- `camera/`、`light_source/`、`reconstruction/` 与 `three_axis_cnc/` 包含组件级说明，是后续扩展运行手册的良好切入点。
- `i18n/` 已预留用于语言特定文档。

## 故障排查

- **Conda 解析问题：** 更新 Conda、检查 channel 顺序，并重试环境创建。
- **Kernel 不匹配：** 确认 Jupyter 使用的是 `lensless` 环境。
- **OpenCV/SIFT 编译错误：** 安装 OpenCV contrib 模块并确认 `opencv2/xfeatures2d.hpp` 可用。
- **Notebook 文件不存在：** 核对预期数据集与 notebook 相对路径。
- **Matcher 未读取到图像：** 确认 `notebooks/multiple_match/all/` 存在且包含有效图像文件。

## 路线图

- 在 `camera/`、`light_source/`、`reconstruction/` 与 `three_axis_cnc/` 中扩展模块级运行手册。
- 文档化数据集契约，并提供可复现的样例数据引用。
- 为主要 notebook 流程补充脚本包装。
- 为重建和匹配输出添加校验检查。
- 完成 `i18n/` 下的多语言 README 文件。

## 参与贡献

欢迎协作与贡献。

- 提交 issue 参与讨论。
- 针对范围内的文档或实验改动提交 pull request。
- 在进行较大硬件与协议级修改前先联系维护者。

## 贡献指南

1. Fork 该仓库。
2. 创建功能分支。
3. 保持改动范围清晰并做好文档化（尤其是 notebook）。
4. 提交 pull request，说明动机、方法和任何验证记录。

## 许可证

当前仓库根目录尚未提供许可证文件。

**假设/待处理：** 请添加 `LICENSE` 文件，并使用准确的 SPDX 标识更新本节内容。

## 联系

若有进一步咨询或合作意向，请联系 `contact@lazealoptix.com`。


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
