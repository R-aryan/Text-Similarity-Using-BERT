import torch

from services.text_similarity.settings import Settings


class BERTDataset:
    def __init__(self, sentence_1, sentence_2, targets):
        self.settings = Settings
        self.sentence_1 = sentence_1
        self.sentence_2 = sentence_2
        self.targets = targets
        assert len(self.sentence_1) == len(self.sentence_2) == len(self.targets)

    def __len__(self):
        return len(self.sentence_1)

    def __getitem__(self, item):
        s1 = self.sentence_1[item]
        s2 = self.sentence_2[item]
        target = self.targets[item]

        inputs = self.settings.tokenizer.encode_plus(
            s1, s2,
            add_special_tokens=True,
            max_length=self.settings.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': torch.tensor(ids),
            'attention_mask': torch.tensor(mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'targets': torch.tensor(target)
        }

