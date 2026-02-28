[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


> 🌐 **多言語対応状況:** `i18n/` は言語別 README ファイル用に確保されています。リンク先のローカライズ文書は計画中/作成中です。

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

*個人利用向けプロトタイプ（左）と機関利用向けプロトタイプ（右）*

## 概要

Lazeal OptiX は、革新的なヘルスケア技術プロジェクトです。プロジェクトの中核は、自宅にいながら高度な診断を利用者に提供するデバイスの開発にあります。先進的な顕微鏡技術と生化学解析技術を活用し、さまざまな健康課題の早期検知を促進することで、医療アウトカムの改善を目指しています。

Lazeal OptiX プロジェクトは、苦しみを減らし、健康診断へのアクセスをより広くするという取り組みから生まれました。個人が自分の健康を主体的に管理できるツールを提供することで、より健康な社会づくりに貢献したいと考えています。

このリポジトリは現在、研究プロトタイプ志向かつノートブック中心の構成です。実装の詳細や実験の多くは `notebooks/` 配下の Jupyter notebooks で管理されています。

### 要点

| 項目 | 現在の状況 |
|---|---|
| プロジェクト成熟度 | 研究プロトタイプ |
| 主な実行モデル | Jupyter ノートブックワークフロー |
| 主な実験領域 | 再構成、光源位置推定、複数画像マッチング |
| ルートでのパッケージング/CI | 現時点では未定義 |
| 多言語ドキュメント | `i18n/` ディレクトリのひな形あり |

## 特徴

1. **高度顕微鏡技術:** 詳細解析のために先進的な顕微鏡技術を活用。
2. **生化学解析:** 深い生化学解析により、さまざまな健康指標の検出を実現。
3. **ユーザーフレンドリー:** 家庭利用を前提に、シンプルで利用しやすい UI を提供。
4. **コンパクトかつ手頃:** Lazeal OptiX は小型で価格も手頃であり、日常ユーザーに高度診断を届けます。
5. **レンズレス再構成ワークフロー:** ノートブックベースの計算イメージングおよび再構成パイプライン。
6. **光源位置推定実験:** 光源位置推定のための最適化ノートブック。
7. **複数画像マッチングユーティリティ:** 特徴点マッチング/アライメントのためのノートブックおよび C++ OpenCV ワークフロー。

## リポジトリ構成

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

### モジュールメモ

- `camera/`: 高解像度サンプル取得のためのカメラ利用に関するスクリプト/リソース。
- `light_source/`: 光源制御および最適化に関するスクリプト/リソース。
- `reconstruction/`: 計算再構成に関するスクリプト/リソース。
- `three_axis_cnc/`: 3 軸 CNC の位置決め/制御に関するスクリプト/リソース。
- `notebooks/`: 実験と手法のための主要な技術ワークスペース。

## Notebooks

`notebooks` ディレクトリには、Lazeal OptiX プロジェクトのさまざまな側面を記録した Jupyter notebooks が含まれています。これらの notebooks には、コード、可視化、そしてプロジェクト手法の詳細な説明が含まれます。プロジェクトを探索し理解するためのインタラクティブな手段として機能します。

### `light_source_location`

`light_source_location` ディレクトリには、光源位置推定に関する notebooks が含まれています。これらの notebooks には、光源位置を高精度に推定するためのアルゴリズムや手法が含まれており、Lazeal OptiX プロジェクトの重要な要素となっています。

### `multiple_match`

`multiple_match` ディレクトリには、複数画像またはパターンのマッチングに関する notebooks とスクリプトが含まれています。この領域では、レンズレス撮像システムから高解像度画像を再構成するために必要な、画像の正確なマッチングと位置合わせを実現する複雑なアルゴリズムを扱います。

### `reconstruction`

`reconstruction` ディレクトリには、Lazeal OptiX デバイスで取得した画像の再構成に関する notebooks が含まれています。これらの notebooks では、レンズレス撮像システムから高解像度画像を再構成するために使われる高度な計算技術を記録しています。

## 前提条件

- OS: 現在の notebook と OpenCV ワークフローでは Linux/macOS を推奨。
- Python: 提供されている環境ファイルは **Python 3.7** を対象。
- Conda: 文書化された `lensless` 環境を再現するために必要。
- Jupyter Notebook/Lab。
- `multiple_match.cpp` 用の任意 C++ ツールチェーン:
  - C++17 対応の `g++`。
  - contrib modules を含む OpenCV 4.x（`opencv2/xfeatures2d.hpp` / SIFT）。

## インストール

### 1) Clone

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) notebook 環境を作成（推奨）

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Jupyter を起動

```bash
jupyter notebook
```

## 使い方

このリポジトリは主に notebooks を開き、セルを順番に実行して利用します。

### 再構成トラック

- データセット準備には `notebooks/reconstruction/dataset_prep.ipynb` を開いてください。
- 再構成/学習実験には `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` を開いてください。

### 光源位置推定トラック

- `notebooks/light_source_location/` 配下の notebooks を開いてください。

### 複数マッチングトラック

- `notebooks/multiple_match/` 配下の notebooks を開いてください。
- 任意の C++ ユーティリティ: `notebooks/multiple_match/multiple_match.cpp`。

## 設定

### Conda environment

主要な環境定義ファイル:

- `notebooks/reconstruction/lensless.yaml`

このファイルから読み取れる主な依存関係シグナル:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- notebooks 内の `opencv` 周辺のコンピュータビジョンワークフロー依存関係

### データとパス

- **前提:** notebooks は、リポジトリルートで一元定義されていないローカルのデータセット/ファイルを前提としています。
- **前提:** C++ マッチングユーティリティは、（実行パスからの相対で）グレースケール読み込み可能な画像を含む `all/` ディレクトリを想定しています。

ローカル環境が異なる場合は、notebook のパス設定セルと C++ 入力ディレクトリを適宜更新してください。

## 例

### マッチングユーティリティを実行（例）

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

期待される挙動:

- `all/` から画像を読み込み
- 画像間で連鎖的な SIFT ベースマッチングを計算
- `result_<timestamp>.png` の形式で出力画像を書き出し

### 特定 notebook を起動

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## 開発メモ

- 現在、リポジトリルートにパッケージ定義（`pyproject.toml`、`requirements.txt`、`setup.py`）および CI/テストハーネスはありません。
- 実験優先の運用で、アルゴリズムの多くは notebooks が一次情報です。
- `camera/`、`light_source/`、`reconstruction/`、`three_axis_cnc/` は現在高レベルのモジュール説明が中心で、今後 runbook の拡充が可能です。
- `i18n/` は存在し、多言語 README バリアント用に確保されています。

## トラブルシューティング

- **Conda solve issues:** Conda を更新してから環境作成を再試行してください。
- **Kernel mismatch in notebooks:** 必要に応じて、アクティブカーネルが `lensless` と一致していることを確認してください。
- **OpenCV/SIFT compile errors:** OpenCV contrib modules を導入し、`opencv2/xfeatures2d.hpp` が利用可能か確認してください。
- **Notebook file-not-found errors:** notebook セルが想定するデータセットパスと相対ディレクトリを確認してください。
- **C++ matcher reads no images:** `notebooks/multiple_match/all/` が存在し、有効な画像ファイルを含むことを確認してください。

## ロードマップ

- `camera/`、`light_source/`、`reconstruction/`、`three_axis_cnc/` のモジュール別 runbook を拡充。
- データセット契約を文書化し、再現可能なサンプルデータ参照先を提供。
- 主要 notebook パイプライン向けに再現可能なスクリプトを追加。
- 再構成およびマッチング出力向けのテスト/検証チェックを追加。
- `i18n/` 配下の多言語 README を完成。

## 参加方法

コラボレーションとコントリビューションを歓迎します。Lazeal OptiX プロジェクトへの参加に関心がある場合は、issue や pull request を送るか、直接ご連絡ください。

## Contributing

1. リポジトリをフォークします。
2. feature ブランチを作成します。
3. 変更範囲を限定し、ドキュメント化してください（特に notebooks）。
4. 動機・手法・検証内容を説明した pull request を作成します。

ハードウェアやプロトコルに大きな変更を加える場合は、事前に issue を立てて方向性をそろえることを推奨します。

## サポート

このリポジトリでは現在、専用の資金提供/スポンサー情報は明示されていません。

今後変更がある場合は、既存の技術ドキュメントを削除せず、このセクションにスポンサーおよび寄付の詳細を追記してください。

## ライセンス

現在、リポジトリルートにライセンスファイルは存在しません。

**前提/必要な対応:** `LICENSE` ファイルを追加し、このセクションを正確な SPDX 識別子で更新してください。

## 連絡先

お問い合わせやコラボレーションのご相談は、`contact@lazealoptix.com` までご連絡ください。
