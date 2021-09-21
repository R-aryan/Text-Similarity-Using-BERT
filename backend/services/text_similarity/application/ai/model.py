import torch.nn as nn
from transformers import BertModel

from services.text_similarity.settings import Settings


class BERTClassifier(nn.Module):
    def __init__(self, freeze_params=False):
        super(BERTClassifier, self).__init__()
        self.settings = Settings
        self.bert = BertModel.from_pretrained(self.settings.checkpoint, return_dict=False)

        # adding custom layers according to the problem statement
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.settings.input_dim, self.settings.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.settings.hidden_dim, self.settings.output_dim)
        # )

        if not freeze_params:
            # freeze all the parameters
            for param in self.bert.parameters():
                param.requires_grad = False

        self.bert_drop = nn.Dropout(self.settings.dropout)
        self.out = nn.Linear(self.settings.input_dim, self.settings.output_dim)

    def forward(self, ids, mask, token_type_ids):
        o1, o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        bo = self.bert_drop(o2)
        output = self.out(bo)

        return output
