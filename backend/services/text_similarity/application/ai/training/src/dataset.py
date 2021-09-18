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


