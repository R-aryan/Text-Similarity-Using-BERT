from services.text_similarity.application.ai.model import BERTClassifier
from services.text_similarity.application.ai.training.src.engine import Engine
from services.text_similarity.application.ai.training.src.preprocess import Preprocess
from services.text_similarity.settings import Settings


from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader


class Train:
    def __init__(self):
        # initialize required class
        self.settings = Settings
        self.engine = Engine()
        self.preprocess = Preprocess()

        # initialize required variables
        self.bert_text_model = None
        self.optimizer = None
        self.scheduler = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.total_steps = None
        self.param_optimizer = None
        self.optimizer_parameters = None
        self.total_steps = None
        self.model_config = None

    def __initialize(self):
        # Instantiate Bert Classifier
        self.bert_text_model = BERTClassifier()
        self.bert_text_model.to(self.settings.DEVICE)
        self.__optimizer_params()

        # Create the optimizer
        self.optimizer = AdamW(self.optimizer_parameters,
                               lr=5e-5,  # Default learning rate
                               eps=1e-8  # Default epsilon value
                               )

        # Set up the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value
                                                         num_training_steps=self.total_steps)

    def __optimizer_params(self):
        self.param_optimizer = list(self.bert_text_model.named_parameters())
        self.optimizer_parameters = [
            {
                "params": [
                    p for n, p in self.param_optimizer if not any(nd in n for nd in self.settings.no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in self.param_optimizer if any(nd in n for nd in self.settings.no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]


