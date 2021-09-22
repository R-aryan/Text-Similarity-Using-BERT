from injector import Module, singleton

from common.logging.console_logger import ConsoleLogger
from services.text_similarity.application.ai.inference.prediction import PredictionManager
from services.text_similarity.application.ai.training.src.preprocess import Preprocess
from services.text_similarity.settings import Settings


class Configuration(Module):
    def configure(self, binder):
        logger = ConsoleLogger(filename=Settings.LOGS_DIRECTORY)
        binder.bind(ConsoleLogger, to=logger, scope=singleton)
        binder.bind(PredictionManager, to=PredictionManager(preprocess=Preprocess(), logger=logger), scope=singleton)
