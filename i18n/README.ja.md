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

> 🌐 **多言語対応状況:** `i18n/` が存在し、言語別 README ファイル用に確保されています。リンク済みのローカライズ文書は計画中または作成中です。

## ✨ 概要

| Focus | Location |
|---|---|
| コアワークフロー | `notebooks/` |
| 環境定義 | `notebooks/reconstruction/lensless.yaml` |
| コンポーネントノート | `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/` |
| 主要なドキュメント | `i18n/README.*.md` |

<table width="100%">
  <tr>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_individual.jpg" alt="個人向けプロトタイプ" style="width: 90%" />
    </td>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_institute.png" alt="機関向けプロトタイプ" style="width: 90%" />
    </td>
  </tr>
</table>

*左: 個人向けプロトタイプ、右: 機関向けプロトタイプ*

## Overview

Lazeal OptiX は、ヘルスケア関連の診断分野におけるレンズレス撮像ワークフロー向けに設計された研究/プロトタイププロジェクトです。本リポジトリは現在ノートブック中心かつ実験的な構成であり、制約のある環境でも高度な診断アプローチを使いやすくすることを目的としています。

主要なコンセプトは以下のとおりです。

- レンズレス画像再構成
- 光源の位置推定
- 複数画像のマッチングとアラインメント

本リポジトリは主として `notebooks/` 配下の Jupyter ノートブックで管理され、モジュールごとのコンテキストは各サブディレクトリに格納されています。

### リポジトリステータス

| 領域 | 現在の状態 |
|---|---|
| プロジェクト成熟度 | 研究プロトタイプ |
| 主な実行モデル | Jupyter ノートブックワークフロー |
| 主な実験領域 | 再構成、光源位置推定、複数画像マッチング |
| ルートでのパッケージ/CI | 現時点では未定義 |
| 多言語ドキュメント | `i18n/` ディレクトリのスケルトンが存在 |

## Features

1. **Advanced Microscopy Concepts**: 詳細解析に必要な高度な光学設計と画像取得パターン。
2. **Biochemical / Diagnostic Context**: 健康指標検出を想定した実験ワークフロー。
3. **Home-friendly Direction**: 実用導入しやすい設計と、現場での運用性を重視。
4. **Laptop-first Experience**: ノートブックを主要な実行経路として採用。
5. **Lensless Reconstruction Utilities**: 高解像度再構成のための計算パイプライン。
6. **Light Source Localization Tools**: 光源位置推定と幾何キャリブレーションの実験。
7. **Multiple Image Matching**: SIFT ベースのマッチング、チェイン処理、位置合わせユーティリティ。

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

- `camera/`: 高解像度サンプル取得向けのカメラ利用スクリプト/リソース。
- `light_source/`: 光源制御・最適化向けのスクリプト/リソース。
- `reconstruction/`: 計算再構成向けのスクリプト/リソース。
- `three_axis_cnc/`: 3 軸 CNC の位置決め・制御向けスクリプト/リソース。
- `notebooks/`: 実験と手法の主要技術ワークスペース。

## Notebooks

`notebooks` ディレクトリには、主要実験手法を説明する Jupyter ノートブックが含まれています。これらのノートブックでは、コード、可視化、方法ノートを領域ごとに提供しています。

### `light_source_location`

光源位置推定に関するノートブックを含みます。これらの手法は、光源幾何キャリブレーションと再構成精度の維持を支援します。

### `multiple_match`

画像/パターンのマッチングとアラインメントを扱うノートブックとスクリプトを含み、堅牢なレジストレーションワークフローを支援します。

### `reconstruction`

取得画像からの再構成に関するノートブックを含み、前処理と実験スクリプトを含みます。

## Prerequisites

- OS: 現在の Conda および OpenCV ワークフローでは Linux/macOS を推奨。
- Python: 環境定義は **Python 3.7** を対象。
- Conda: 文書化された `lensless` 環境を再現するために必要。
- Jupyter Notebook/Lab。
- `multiple_match.cpp` 用の任意 C++ ツールチェーン:
  - C++17 に対応した `g++`
  - OpenCV 4.x の contrib モジュール（`opencv2/xfeatures2d.hpp` / SIFT）

## Installation

### 1) クローン

```bash

git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) notebook 用環境の作成

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Jupyter の起動

```bash
jupyter notebook
```

## Usage

このリポジトリは主にノートブックを開いて、記載順にセルを実行して利用します。

### Reconstruction track

- `notebooks/reconstruction/dataset_prep.ipynb` を開き、データセットを準備します。
- `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` を開き、再構成/学習実験を実行します。

### Light source localization track

- `notebooks/light_source_location/` 以下のノートブックを開きます。

### Multiple match track

- `notebooks/multiple_match/` 以下のノートブックを開きます。
- 補助ユーティリティ: `notebooks/multiple_match/multiple_match.cpp`

## Configuration

### Conda environment

主要な環境定義:

- `notebooks/reconstruction/lensless.yaml`

主な依存関係:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- ノートブック内の `opencv` 関連コンピュータビジョンワークフロー依存

### Data and paths

- **前提:** データセットはローカルに置かれており、リポジトリルートで一元管理されていません。
- **前提:** C++ のマッチングユーティリティは、実行時パスからの相対位置にある `all/` ディレクトリ内に、グレースケール読み取り可能な画像が入っていることを想定しています。

ローカル環境が異なる場合は、ノートブックのパス参照セルと C++ の入力ディレクトリを適切に更新してください。

## Examples

### マッチングユーティリティの実行

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

想定される挙動:

- `all/` から画像を読み込む
- SIFT ベースの連鎖マッチングを複数画像間で計算する
- `result_<timestamp>.png` のような出力画像を書き出す

### 特定のノートブックを起動

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## Development Notes

- リポジトリルートにはパッケージ用のマニフェスト（`pyproject.toml`、`requirements.txt`、`setup.py`）や CI/test ハーネスは現在ありません。
- 実験優先で、ノートブックが現行アルゴリズムの情報源（source-of-truth）です。
- `camera/`、`light_source/`、`reconstruction/`、`three_axis_cnc/` にはコンポーネントレベルの説明があり、今後の運用手順書（runbook）追加の拡張ポイントになります。
- `i18n/` は言語別ドキュメント用に準備済みです。

## Troubleshooting

- **Conda の解決失敗:** Conda を更新し、チャネルの順序を確認して環境作成を再実行してください。
- **ノートブックのカーネル不一致:** Jupyter が `lensless` 環境を使っているか確認してください。
- **OpenCV/SIFT のビルドエラー:** OpenCV contrib モジュールをインストールし、`opencv2/xfeatures2d.hpp` が利用可能であることを確認してください。
- **ノートブックのファイル未検出エラー:** 想定するデータセットとノートブック相対パスを確認してください。
- **Matcher が画像を読めない:** `notebooks/multiple_match/all/` が存在し、有効な画像ファイルを含んでいるか確認してください。

## Roadmap

- `camera/`、`light_source/`、`reconstruction/`、`three_axis_cnc/` のモジュール別 runbook を拡張。
- データセット仕様を文書化し、再現可能なサンプルデータ参照を提供。
- 主要なノートブックパイプライン用にスクリプトラッパーを追加。
- 再構成およびマッチング結果の検証チェックを追加。
- `i18n/` 下で多言語 README を完成。

## Getting Involved

コラボレーションや貢献を歓迎します。

- 議論のために issue を立ててください。
- 範囲の限定されたドキュメント修正や実験変更は pull request で提出してください。
- 大規模なハードウェア/プロトコル変更前に、事前にメンテナと相談してください。

## Contributing

1. リポジトリをフォークする。
2. フィーチャーブランチを作成する。
3. 変更範囲を絞って記録する（特にノートブックは必須）。
4. 目的、方法、検証ノートを記載した pull request を作成する。

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License

現在、リポジトリルートにはライセンスファイルがありません。

**前提/対応が必要:** `LICENSE` ファイルを追加し、このセクションを正確な SPDX 識別子に更新してください。

## Contact

追加の問い合わせや共同作業の依頼は、`contact@lazealoptix.com` までご連絡ください。
