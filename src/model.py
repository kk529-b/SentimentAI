import torch.nn as nn
from transformers import BertModel


class SentimentClassifier(nn.Module):
    """
    BERTの [CLS] ベクトルを取り出して、2クラス分類（Positive/Negative）するモデル
    """
    def __init__(self, bert_model_name: str, hidden_dim: int = 768, output_dim: int = 2):
        super().__init__()
        # 事前学習済みBERTを読み込む
        self.bert = BertModel.from_pretrained(bert_model_name)

        # 過学習対策のDropout
        self.dropout = nn.Dropout(p=0.3)

        # [CLS]ベクトル(768次元) → 2クラス への線形変換
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        """
        input_ids, attention_mask をBERTに入れて、最後の層の出力から [CLS] を使う
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # last_hidden_state: [batch, seq_len, hidden]
        # 先頭トークン（[CLS]）だけ取り出す
        cls_output = outputs.last_hidden_state[:, 0, :]

        x = self.dropout(cls_output)
        x = self.classifier(x)  # [batch, 2]
        return x