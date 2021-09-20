import pandas as pd
from sklearn import model_selection

from services.text_similarity.application.ai.model import BERTClassifier
from services.text_similarity.application.ai.training.src import utils
from services.text_similarity.application.ai.training.src.dataset import BERTDataset
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
        self.early_stopping = utils.EarlyStopping(patience=self.settings.patience,
                                                  mode=self.settings.mode)

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

    def __create_data_loaders(self, sentence1, sentence2, targets, batch_size, num_workers):
        dataset = BERTDataset(sentence_1=sentence1,
                              sentence_2=sentence2,
                              targets=targets)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        return data_loader

    def __load_data(self, csv_data_path):
        df = pd.read_csv(csv_data_path).dropna().reset_index(drop=True)
        df_train, df_valid = model_selection.train_test_split(
            df,
            random_state=self.settings.seed_value,
            test_size=self.settings.test_size,
            stratify=df.is_duplicate.values

        )

        df_train = df_train.reset_index(drop=True)
        df_valid = df_valid.reset_index(drop=True)

        # creating Data Loaders
        # train data loader
        self.train_data_loader = self.__create_data_loaders(sentence1=df_train.question1.values,
                                                            sentence2=df_train.question2.values,
                                                            targets=df_train.is_duplicate.values,
                                                            num_workers=self.settings.TRAIN_NUM_WORKERS)

        # validation data loader
        self.val_data_loader = self.__create_data_loaders(sentence1=df_valid.question1.values,
                                                          sentence2=df_valid.question2.values,
                                                          targets=df_valid.is_duplicate.values,
                                                          num_workers=self.settings.VAL_NUM_WORKERS)

        self.total_steps = int(len(df_train) / self.settings.TRAIN_BATCH_SIZE * self.settings.EPOCHS)

    def train(self):
        pass

