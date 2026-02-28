[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)


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

> 🌐 **Multilingual status:** `i18n/` is present and reserved for language-specific README files. Linked localized documents are planned/in-progress.

## ✨ At a Glance

| Focus | Location |
|---|---|
| Core workflow | `notebooks/` |
| Environment spec | `notebooks/reconstruction/lensless.yaml` |
| Component notes | `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/` |
| Entry docs | `i18n/README.*.md` |

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

*Prototype for Individual Use (left) and Institutional Use (right)*

## Overview

Lazeal OptiX is a research/prototype project for lensless imaging workflows in healthcare-adjacent diagnostics. The repository is currently notebook-centric and experimental, and is intended to make advanced diagnostic approaches more accessible in constrained settings.

Core ideas include:

- lensless image reconstruction,
- light source localization,
- multiple-image matching and alignment.

The repository is primarily maintained through Jupyter notebooks under `notebooks/`, with module-specific context stored in dedicated directories.

### Repository Status Snapshot

| Area | Current Status |
|---|---|
| Project maturity | Research prototype |
| Primary execution model | Jupyter notebook workflows |
| Main experiment domains | Reconstruction, light source localization, multiple-image matching |
| Packaging/CI at root | Not currently declared |
| Multilingual docs | `i18n/` directory scaffold exists |

## Features

1. **Advanced Microscopy Concepts**: advanced optics and image capture patterns for detailed analysis.
2. **Biochemical / Diagnostic Context**: experimental workflows aimed at health-indicator detection.
3. **Home-friendly Direction**: designed for accessible use and practical deployment.
4. **Laptop-first Experience**: notebooks provide the primary execution path.
5. **Lensless Reconstruction Utilities**: computational pipelines for high-resolution reconstruction.
6. **Light Source Localization Tools**: experiments on source localization and geometry calibration.
7. **Multiple Image Matching**: SIFT-based matching, chaining, and alignment utilities.

## Project Structure

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

### Module Notes

- `camera/`: scripts/resources related to camera usage for high-resolution sample capture.
- `light_source/`: scripts/resources for light source control and optimization.
- `reconstruction/`: scripts/resources for computational reconstruction.
- `three_axis_cnc/`: scripts/resources for three-axis CNC positioning/control.
- `notebooks/`: primary technical workspace for experiments and methods.

## Notebooks

The `notebooks` directory contains Jupyter notebooks that document the core experimental methods. These notebooks provide code, visualizations, and method notes for each area.

### `light_source_location`

Contains notebooks related to estimation of light source locations. These methods support source geometry calibration and reconstruction fidelity.

### `multiple_match`

Contains notebooks and scripts for image/pattern matching and alignment to support robust registration workflows.

### `reconstruction`

Contains notebooks related to reconstruction from captured images, including preprocessing and experiment scripts.

## Prerequisites

- OS: Linux/macOS recommended for current Conda and OpenCV workflows.
- Python: environment targets **Python 3.7**.
- Conda: needed to reproduce the documented `lensless` environment.
- Jupyter Notebook/Lab.
- Optional C++ toolchain for `multiple_match.cpp`:
  - `g++` with C++17 support.
  - OpenCV 4.x with contrib modules (`opencv2/xfeatures2d.hpp` / SIFT).

## Installation

### 1) Clone

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) Create notebook environment

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Start Jupyter

```bash
jupyter notebook
```

## Usage

This repository is primarily used by opening notebooks and running cells in documented order.

### Reconstruction track

- Open `notebooks/reconstruction/dataset_prep.ipynb` for dataset preparation.
- Open `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` for reconstruction/training experiments.

### Light source localization track

- Open notebooks under `notebooks/light_source_location/`.

### Multiple match track

- Open notebooks under `notebooks/multiple_match/`.
- Optional utility: `notebooks/multiple_match/multiple_match.cpp`.

## Configuration

### Conda environment

Primary environment specification:

- `notebooks/reconstruction/lensless.yaml`

Notable dependencies include:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- `opencv`-adjacent computer vision workflow dependencies in notebooks

### Data and paths

- **Assumption:** datasets are local and are not centrally declared at repository root.
- **Assumption:** the C++ matching utility expects an `all/` directory (relative to its execution path) containing grayscale-readable images.

If your local setup differs, update notebook path cells and the C++ input directory accordingly.

## Examples

### Run the matching utility

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

Expected behavior:

- Reads images from `all/`
- Computes chained SIFT-based matches across images
- Writes an output image like `result_<timestamp>.png`

### Launch a specific notebook

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## Development Notes

- No root-level packaging manifest (`pyproject.toml`, `requirements.txt`, `setup.py`) or CI/test harness is currently present.
- Work is experiment-first; notebooks are the source-of-truth for current algorithms.
- `camera/`, `light_source/`, `reconstruction/`, and `three_axis_cnc/` contain component-level descriptions and are good extension points for runbooks.
- `i18n/` is prepped for language-specific documentation.

## Troubleshooting

- **Conda solve issues:** update Conda, verify channel ordering, and retry environment creation.
- **Kernel mismatch in notebooks:** confirm Jupyter is using the `lensless` environment.
- **OpenCV/SIFT compile errors:** install OpenCV contrib modules and validate `opencv2/xfeatures2d.hpp` availability.
- **Notebook file-not-found errors:** verify expected datasets and notebook-relative paths.
- **Matcher reads no images:** ensure `notebooks/multiple_match/all/` exists with valid image files.

## Roadmap

- Expand module-level runbooks in `camera/`, `light_source/`, `reconstruction/`, and `three_axis_cnc/`.
- Document dataset contracts and provide reproducible sample-data references.
- Add script wrappers for major notebook pipelines.
- Add validation checks for reconstruction and matching outputs.
- Complete multilingual README files under `i18n/`.

## Getting Involved

We welcome collaboration and contributions.

- Open an issue for discussion.
- Submit a pull request for scoped documentation or experimental changes.
- Contact maintainers for hardware and protocol-level changes before large refactors.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Keep changes scoped and documented (especially for notebooks).
4. Open a pull request with motivation, method, and any validation notes.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License

No license file is currently present in the repository root.

**Assumption/Action needed:** add a `LICENSE` file and update this section with the exact SPDX identifier.

## Contact

For further inquiries or collaboration interests, please reach out at `contact@lazealoptix.com`.
