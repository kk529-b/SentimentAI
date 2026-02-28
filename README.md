# SentimentAI — 感情分析（BERT / IMDB）

本プロジェクトは、IMDBの映画レビュー（英語）データを用いた二値感情分類（Positive / Negative）を行う学習・推論用のリポジトリです。
Colabで作成したモデルをローカルで再現できるよう、モジュール化と簡易CLIを備えています。

**主な特徴**
- モデル: `bert-base-uncased`（Hugging Face Transformers）
- フレームワーク: PyTorch
- 用途: 学習 (train) / 推論 (predict)

---

**目次**
- 概要
- 要件
- セットアップ
- 使い方（学習・推論）
- ファイル構成
- 注意事項

---

## 概要

IMDBレビューを使った二値分類タスクのサンプル実装です。学習済みモデルとトークナイザを `outputs/` 以下に保存し、`predict.py` で単一文章の感情予測が可能です。

## 要件

- Python 3.8〜3.11 を推奨
- PyTorch
- transformers
- その他: `requirements.txt` を参照してください

---

## セットアップ

1. 仮想環境を作成・有効化

Windows (PowerShell):

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
```

Unix/macOS (bash):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. 依存関係をインストール

```bash
pip install -r requirements.txt
```

---

## 使い方

### 学習 (train)

学習は `train.py` を使います。基本的な実行例:

```bash
python train.py --epochs 3 --batch_size 16 --output_dir outputs
```

主要なオプションはスクリプト内のヘルプを参照してください。

学習後、モデルは `outputs/model.pt`、トークナイザ設定は `outputs/tokenizer.json` や `outputs/tokenizer_config.json` に保存されます。

### 推論 (predict)

単一文章の感情予測は `predict.py` を使用します。例:

```bash
python predict.py --model outputs/model.pt --text "This movie was fantastic!"
```

戻り値はラベル（Positive/Negative）と確信度です。

---

## ファイル構成（主なもの）

- `train.py` : 学習用スクリプト
- `predict.py` : 推論用スクリプト（CLI）
- `model.py` : モデル定義
- `data.py` : データ読み込み・前処理
- `outputs/` : 学習済みモデル・トークナイザ等の出力ディレクトリ
	- `model.pt`
	- `tokenizer.json`
	- `tokenizer_config.json`
- `requirements.txt` : 依存パッケージ

---

## 注意事項

- GPUで実行する場合はPyTorchのCUDA対応ビルドをインストールしてください。
- Python / ライブラリのバージョン差で動作が変わることがあります。`requirements.txt` を基準に環境を合わせてください。
- 大規模な学習は計算資源を大きく消費します。サンプル設定は軽めに設計されています。

---

## 追加情報 / 次のステップ

- データをカスタムに差し替える場合は `data.py` を編集してください。
- 評価指標やモデル保存・ロードの挙動をカスタマイズ可能です。

---

## 問い合わせ

問題や改善提案があれば Issue を作成してください。

---

© このプロジェクト