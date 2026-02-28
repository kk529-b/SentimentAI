import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import prepare_dataloaders
from model import SentimentClassifier


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """1エポック分の学習を行い、平均LossとAccuracyを返す"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="学習", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """検証（評価）を行い、平均LossとAccuracyを返す"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="評価", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)

    # Notebookの「短い文章だけ使う」条件に合わせる
    parser.add_argument("--max_tokens", type=int, default=100)

    # Tokenize時のpadding/truncation長
    parser.add_argument("--max_length", type=int, default=256)

    # 出力先（学習済みモデルなどを保存）
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, train_loader, test_loader, n_train, n_test = prepare_dataloaders(
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    print(f"フィルタ後のtrain件数: {n_train}")
    print(f"フィルタ後のtest件数 : {n_test}")
    print(f"使用デバイス: {device}")

    model = SentimentClassifier(bert_model_name=args.model_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        eval_loss, eval_acc = evaluate(model, test_loader, loss_fn, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Eval  loss: {eval_loss:.4f}, acc: {eval_acc:.4f}"
        )

    # 学習済みモデルとTokenizerを保存
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    tokenizer.save_pretrained(args.output_dir)

    # 学習時の設定も保存（再現性のため）
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"保存完了: {args.output_dir}")


if __name__ == "__main__":
    main()