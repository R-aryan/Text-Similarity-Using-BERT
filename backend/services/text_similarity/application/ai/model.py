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

    def print_model_details(self):
        # Get all of the model's parameters as a list of tuples.
        params = list(self.bert.named_parameters())

        print('The BERT Base Uncased Model Has {:} different named parameters.\n'.format(len(params)))

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')

        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
