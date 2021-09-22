import torch
from injector import inject

from common.logging.console_logger import ConsoleLogger
from services.text_similarity.application.ai.model import BERTClassifier
from services.text_similarity.application.ai.training.src.preprocess import Preprocess
from services.text_similarity.settings import Settings


class Prediction:
    @inject
    def __init__(self, preprocess: Preprocess, logger: ConsoleLogger):
        self.settings = Settings
        self.preprocess = preprocess
        self.logger = logger

        self.__model = None

    def __load_model(self):
        try:
            # print("-------Loading Bert Base Model------")
            self.logger.info(message="Loading Bert Base Uncased Model.")
            self.__model = BERTClassifier()
            # print("-------Bert Base Model Successfully Loaded---- \n\n")
            self.logger.info(message="Bert Base Model Successfully Loaded.")

            # print('Loading Model Weights----!!')
            self.logger.info(message="Loading Model trained Weights.")
            self.__model.load_state_dict(torch.load(self.settings.WEIGHTS_PATH,
                                                    map_location=torch.device(self.settings.DEVICE)))
            self.__model.to(self.settings.DEVICE)
            self.__model.eval()
            self.logger.info(message="Model Weights loaded Successfully--!!")

        except BaseException as ex:
            # print("Following Exception Occurred---!! ", str(ex))
            self.logger.error(message="Exception Occurred while loading model---!! " + str(ex))
