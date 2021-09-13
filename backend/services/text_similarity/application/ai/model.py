from torch import nn
from transformers import AutoModelForSequenceClassification

from services.text_similarity.settings import Settings


class BERTClassifier(nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.settings = Settings
        self.model = AutoModelForSequenceClassification.from_pretrained(self.settings.checkpoint)
