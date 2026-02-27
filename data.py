from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def preprocess(example: dict) -> dict:
    """
    テキストの前処理：
    - 改行やHTMLの <br> などを除去して、学習しやすい形に整える
    """
    text = example["text"]
    text = text.replace("\n", "")
    text = text.replace("<br>", "")
    text = text.replace("<br />", "")
    return {"cleaned_texts": text}


def build_tokenizer(model_name: str):
    """指定した事前学習モデル名からTokenizerを読み込む"""
    return AutoTokenizer.from_pretrained(model_name)


def is_short_enough_factory(tokenizer, max_tokens: int):
    """
    文章をトークン化したときの長さが max_tokens 以下かどうかでフィルタする関数を作る
    （Notebookの「100トークン以下に絞る」と同じ意図）
    """
    def is_short_enough(example: dict) -> bool:
        return len(tokenizer.tokenize(example["cleaned_texts"])) <= max_tokens
    return is_short_enough


def tokenizer_function_factory(tokenizer, max_length: int):
    """
    cleaned_texts を Tokenize して input_ids / attention_mask を作る関数を作る
    - padding: max_length まで埋める
    - truncation: max_length を超えたら切る
    """
    def tokenizer_function(example: dict) -> dict:
        return tokenizer(
            example["cleaned_texts"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
    return tokenizer_function


def prepare_dataloaders(
    model_name: str = "bert-base-uncased",
    max_tokens: int = 100,
    max_length: int = 256,
    batch_size: int = 32,
):
    """
    IMDBデータセットを読み込み、前処理 → フィルタ → Tokenize → DataLoader作成まで行う
    戻り値：
    - tokenizer
    - train_loader, test_loader
    - train/testのデータ件数（フィルタ後）
    """
    dataset = load_dataset("imdb")

    # テキストの前処理
    dataset["train"] = dataset["train"].map(preprocess)
    dataset["test"] = dataset["test"].map(preprocess)

    tokenizer = build_tokenizer(model_name)

    # トークン長でフィルタ（短い文章のみ残す）
    is_short_enough = is_short_enough_factory(tokenizer, max_tokens=max_tokens)
    train_ds = dataset["train"].filter(is_short_enough)
    test_ds = dataset["test"].filter(is_short_enough)

    # Tokenize（input_ids/attention_maskを作る）
    tokenizer_function = tokenizer_function_factory(tokenizer, max_length=max_length)
    train_ds = train_ds.map(tokenizer_function)
    test_ds = test_ds.map(tokenizer_function)

    # PyTorchテンソルとして扱えるように列を指定
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # DataLoader化
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return tokenizer, train_loader, test_loader, len(train_ds), len(test_ds)