import argparse
import torch
from transformers import AutoTokenizer

from model import SentimentClassifier


def predict_sentiment(model, tokenizer, text: str, device, max_len: int = 128) -> str:
    """
    1文を入力として、Positive / Negative を返す
    """
    model.eval()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs, dim=1).item()

    return "Positive" if pred == 1 else "Negative"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="outputs")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")  # 学習時と同じにする
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 保存済みTokenizerを読み込む
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # モデルを作って重みをロード
    model = SentimentClassifier(bert_model_name=args.model_name).to(device)
    state = torch.load(f"{args.model_dir}/model.pt", map_location=device)
    model.load_state_dict(state)

    text = input("英語レビューを入力してください: ")
    result = predict_sentiment(model, tokenizer, text, device, max_len=args.max_len)
    print(f"判定結果：{result}")


if __name__ == "__main__":
    main()