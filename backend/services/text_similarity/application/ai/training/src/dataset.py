from services.text_similarity.settings import Settings


class BERTDataset:
    def __init__(self, texts, targets):
        self.settings = Settings
        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        pass
