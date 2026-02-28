[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


> 🌐 **多语言状态：** `i18n/` 目录已存在，并预留用于各语言 README 文件。已链接的本地化文档处于计划中/进行中。

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

*个人使用原型（左）与机构使用原型（右）*

## 概览

Lazeal OptiX 是一个创新型医疗健康技术项目。项目核心是开发一款可让用户在家中舒适环境下完成高级诊断的设备。该设备结合先进显微技术与生化分析方法，目标是促进多类健康问题的早期发现，从而改善整体医疗结果。

Lazeal OptiX 项目源于“减少痛苦、让健康诊断更普惠”的承诺。我们希望通过为个人提供主动管理健康所需的工具，助力构建更健康的社会。

当前仓库以研究/原型阶段为主，且工作流以 Notebook 为中心。多数实现细节与实验都记录在 `notebooks/` 下的 Jupyter notebooks 中。

### 速览

| 领域 | 当前状态 |
|---|---|
| 项目成熟度 | 研究原型 |
| 主要执行模型 | Jupyter notebook 工作流 |
| 主要实验方向 | 重建、光源定位、多图像匹配 |
| 根目录打包/CI | 当前未声明 |
| 多语言文档 | 已有 `i18n/` 目录骨架 |

## 功能特性

1. **高级显微分析：** 利用先进显微技术进行精细化分析。
2. **生化分析：** 深度生化分析可检测多种健康指标。
3. **易用性：** 面向居家场景设计，提供简单且易访问的用户界面。
4. **紧凑且可负担：** Lazeal OptiX 体积紧凑、价格友好，让高级诊断走向日常用户。
5. **无透镜重建工作流：** 基于 Notebook 的计算成像与重建流程。
6. **光源定位实验：** 用于光源位置估计的优化类 Notebook。
7. **多图像匹配工具：** 基于 Notebook 与 C++ OpenCV 的特征匹配/对齐流程。

## 仓库结构

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

### 模块说明

- `camera/`：与高分辨率样本采集相机使用相关的脚本/资源。
- `light_source/`：用于光源控制与优化的脚本/资源。
- `reconstruction/`：用于计算重建的脚本/资源。
- `three_axis_cnc/`：用于三轴 CNC 定位/控制的脚本/资源。
- `notebooks/`：实验方法的主要技术工作区。

## Notebooks

`notebooks` 目录包含记录 Lazeal OptiX 项目多个方面的 Jupyter notebooks。这些 notebook 包含代码、可视化和项目方法论的详细说明，提供了交互式探索和理解项目的方式。

### `light_source_location`

`light_source_location` 目录包含与光源位置估计相关的 notebooks。这些 notebooks 展示了用于准确估计光源位置的算法与方法，这是 Lazeal OptiX 项目的关键环节之一。

### `multiple_match`

`multiple_match` 目录包含与多图像或多模式匹配相关的 notebooks 与脚本。该部分涉及复杂算法，用于精确匹配并对齐图像，这是从无透镜成像系统重建高分辨率图像的必要步骤。

### `reconstruction`

`reconstruction` 目录包含与 Lazeal OptiX 设备采集图像重建相关的 notebooks。这些 notebooks 记录了从无透镜成像系统重建高分辨率图像所采用的高级计算技术。

## 前置要求

- 操作系统：当前 notebook 与 OpenCV 工作流建议使用 Linux/macOS。
- Python：提供的环境文件目标版本为 **Python 3.7**。
- Conda：用于复现文档中的 `lensless` 环境。
- Jupyter Notebook/Lab。
- `multiple_match.cpp` 的可选 C++ 工具链：
  - 支持 C++17 的 `g++`。
  - 含 contrib 模块的 OpenCV 4.x（`opencv2/xfeatures2d.hpp` / SIFT）。

## 安装

### 1) 克隆仓库

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) 创建 notebook 环境（推荐）

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) 启动 Jupyter

```bash
jupyter notebook
```

## 使用方式

本仓库主要通过打开 notebooks 并按顺序运行单元格来使用。

### 重建路线

- 打开 `notebooks/reconstruction/dataset_prep.ipynb` 进行数据集准备。
- 打开 `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` 进行重建/训练实验。

### 光源定位路线

- 打开 `notebooks/light_source_location/` 下的 notebooks。

### 多图像匹配路线

- 打开 `notebooks/multiple_match/` 下的 notebooks。
- 可选 C++ 工具：`notebooks/multiple_match/multiple_match.cpp`。

## 配置

### Conda 环境

主要环境规范文件位于：

- `notebooks/reconstruction/lensless.yaml`

该文件中较关键的依赖信号包括：

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- notebooks 中与 `opencv` 相关的计算机视觉工作流依赖

### 数据与路径

- **假设：** notebooks 依赖本地数据集/文件，且这些内容未在仓库根目录集中声明。
- **假设：** C++ 匹配工具期望存在 `all/` 目录（相对于其执行路径），且其中包含可按灰度读取的图像。

如果你的本地环境与上述假设不同，请相应更新 notebook 中的路径单元以及 C++ 输入目录配置。

## 示例

### 运行匹配工具（示例）

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

预期行为：

- 从 `all/` 读取图像
- 计算跨图像链式 SIFT 匹配
- 输出形如 `result_<timestamp>.png` 的结果图像

### 启动指定 notebook

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## 开发说明

- 仓库当前在根目录尚无打包文件（`pyproject.toml`、`requirements.txt` 或 `setup.py`），也没有根级 CI/测试框架。
- 工作模式以实验优先：多数算法以 notebooks 为事实来源（source-of-truth）。
- `camera/`、`light_source/`、`reconstruction/` 与 `three_axis_cnc/` 当前提供较高层级模块说明，后续可逐步扩展为可执行 runbook。
- `i18n/` 已存在并预留用于多语言 README 变体。

## 故障排查

- **Conda 解析问题：** 更新 Conda 后重试创建环境。
- **Notebook 内核不匹配：** 确保激活内核与所需的 `lensless` 一致。
- **OpenCV/SIFT 编译错误：** 安装 OpenCV contrib 模块，并确认 `opencv2/xfeatures2d.hpp` 可用。
- **Notebook 找不到文件：** 检查数据集路径及各 notebook 单元所期望的相对目录。
- **C++ 匹配器未读到图像：** 确认 `notebooks/multiple_match/all/` 存在且包含有效图像文件。

## 路线图

- 扩展 `camera/`、`light_source/`、`reconstruction/` 和 `three_axis_cnc/` 的模块级 runbook。
- 补充数据集契约说明，并提供可复现样例数据指引。
- 为关键 notebook 流程增加可复现脚本。
- 为重建与匹配输出加入测试/验证检查。
- 完成 `i18n/` 目录下的多语言 README 文件。

## 参与项目

欢迎协作与贡献。如果你有兴趣参与 Lazeal OptiX 项目，欢迎提交 issue 或 pull request，也可以直接联系我们。

## 贡献指南

1. Fork 本仓库。
2. 创建功能分支。
3. 将改动范围控制清晰并补充文档（尤其是 notebooks 相关改动）。
4. 提交 pull request，说明动机、方法与验证结果。

若计划进行较大的硬件/协议变更，建议先开 issue 以便达成一致。

## 支持

当前仓库尚未声明专门的资金支持/赞助元数据。

若后续有变化，应在此处补充赞助与捐赠信息，同时不移除现有技术文档内容。

## 许可证

仓库根目录当前不存在许可证文件。

**假设/需要执行的动作：** 添加 `LICENSE` 文件，并在本节更新精确的 SPDX 标识符。

## 联系方式

如有进一步咨询或合作意向，请联系：`contact@lazealoptix.com`。
