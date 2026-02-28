# 感情分析（BERT / IMDB）

IMDB映画レビュー（英語）を用いて、BERTによる二値感情分類（Positive / Negative）を実装したプロジェクトです。

Google Colab上で構築したモデルを、ローカル環境で再現可能な構成（モジュール分割・CLI実行形式）に整理しました。

---

## 📌 プロジェクト概要

- **データセット**：IMDB Movie Reviews（25,000 train / 25,000 test）
- **タスク**：二値感情分類（Positive / Negative）
- **モデル**：`bert-base-uncased`
- **フレームワーク**：PyTorch / Hugging Face Transformers
- **評価指標**：Accuracy

---

## 🎯 目的

1. 事前学習済み言語モデル（BERT）のファインチューニング理解
2. トークナイズ〜学習〜評価までの一連のパイプライン構築
3. 実験コードを再利用可能な構造にリファクタリング

---

## 🏗 ディレクトリ構成

```
.
├── data/               # データ保存ディレクトリ
├── src/
│   ├── dataset.py      # Dataset定義
│   ├── model.py        # モデル定義
│   ├── train.py        # 学習処理
│   ├── evaluate.py     # 評価処理
│   └── utils.py        # 補助関数
├── main.py             # CLI実行エントリポイント
├── requirements.txt
└── README.md
```

---

## 🛠 実行環境

- Python 3.11 推奨
  > ※ Python 3.13ではtorchが正常に動作しない場合あり
- Windows / macOS 対応
- CPU実行可（GPU推奨）

---

## 🚀 セットアップ手順

1. 仮想環境の作成
   ```bash
   python -m venv .venv
   ```
2. 仮想環境の有効化

   **Windows**
   ```powershell
   .venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   source .venv/bin/activate
   ```
3. 依存ライブラリのインストール
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ 実行方法

- **学習**
  ```bash
  python main.py --mode train
  ```

- **評価**
  ```bash
  python main.py --mode eval
  ```

---

## 📊 モデル性能（例）

- Accuracy：約90%前後
  > ※ 実行環境・乱数シードにより変動します

---

## 🔍 技術的ポイント

- Hugging Face `AutoTokenizer` / `AutoModelForSequenceClassification` を使用
- Padding / Truncation 処理の実装
- AdamW + Linear Scheduler
- GPU / CPU 自動切り替え
- CLI引数によるモード分岐設計

---

## 💡 今後の改善

- ハイパーパラメータ探索
- Early Stopping導入
- 推論専用スクリプト追加
- Webデモ化（Streamlit等）
