from torch import nn
from transformers import AutoModelForSequenceClassification

from services.text_similarity.settings import Settings


class BERTClassifier(nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.settings = Settings
        self.model = AutoModelForSequenceClassification.from_pretrained(self.settings.checkpoint,
                                                                        num_labels=self.settings.num_labels)

        # self.bert_drop = nn.Dropout(self.settings.dropout)
        # self.out = nn.Linear(self.settings.input_dim, self.settings.num_labels)

    # def forward(self, ids, mask, token_type_ids):
    #     o1, o2 = self.bert(
    #         ids,
    #         attention_mask=mask,
    #         token_type_ids=token_type_ids
    #     )
    #
    #     bo = self.bert_drop(o2)
    #     output = self.out(bo)
    #
    #     return output
