import time

import torch
from injector import inject

from common.logging.console_logger import ConsoleLogger
from services.text_similarity.application.ai.model import BERTClassifier
from services.text_similarity.application.ai.training.src.dataset import BERTDataset
from services.text_similarity.application.ai.training.src.preprocess import Preprocess
from services.text_similarity.settings import Settings


class PredictionManager:
    @inject
    def __init__(self, preprocess: Preprocess, logger: ConsoleLogger):
        self.settings = Settings
        self.preprocess = preprocess
        self.logger = logger

        self.__model = None
        self.__load_model()

    def __load_model(self):
        try:
            self.logger.info(message="Loading Bert Base Uncased Model.")
            self.__model = BERTClassifier()
            self.logger.info(message="Bert Base Model Successfully Loaded.")

            self.logger.info(message="Loading Model trained Weights.")
            self.__model.load_state_dict(torch.load(self.settings.WEIGHTS_PATH,
                                                    map_location=torch.device(self.settings.DEVICE)))
            self.__model.to(self.settings.DEVICE)
            self.__model.eval()
            self.logger.info(message="Model Weights loaded Successfully--!!")

        except BaseException as ex:
            self.logger.error(message="Exception Occurred while loading model---!! " + str(ex))

    def preprocessing_for_bert(self, data):
        pass

    def __predict(self, data):
        try:
            self.logger.info(message="Performing prediction on the given data.")
            test_dataset = BERTDataset(
                sentence_1=[data['sentence_1']],
                sentence_2=[data['sentence_2']],
                targets=[self.settings.TARGETS_DEFAULT_KEY]
            )

            with torch.no_grad():
                start_time = time.time()
                data = test_dataset[0]
                b_input_ids = data['input_ids']
                b_attn_mask = data['attention_mask']
                b_token_type_ids = data['token_type_ids']

                # moving tensors to device
                b_input_ids = b_input_ids.to(self.settings.DEVICE, dtype=torch.long).unsqueeze(0)
                b_attn_mask = b_attn_mask.to(self.settings.DEVICE, dtype=torch.long).unsqueeze(0)
                b_token_type_ids = b_token_type_ids.to(self.settings.DEVICE, dtype=torch.long).unsqueeze(0)

                outputs = self.__model(
                    ids=b_input_ids,
                    mask=b_attn_mask,
                    token_type_ids=b_token_type_ids
                )

                result = torch.sigmoid(outputs).cpu().detach().numpy()

                inference_time = round((time.time() - start_time), 3)

                return result[0], inference_time

        except BaseException as ex:
            self.logger.error(message="Exception Occurred while prediction---!! " + str(ex))

    def __map_response(self, output):
        label = 1 if output[0] > self.settings.threshold else 0
        result = self.settings.possible_labels[label]

        return result

    def __format_response(self, data, result, duration):
        return {
            'input': data,
            'response': result,
            'inference time in sec': duration,
            'device': self.settings.DEVICE
        }

    def run_inference(self, data):
        try:
            self.logger.info(message="Data for inference received---!!  " + str(data))
            self.logger.info(message="Running Inference----!!")
            output, prediction_time = self.__predict(data)
            result = self.__map_response(output)
            inference_data = self.__format_response(data=data,
                                                    result=result,
                                                    duration=prediction_time)
            self.logger.info(message="prediction Successful.: Response " + str(inference_data))
            return inference_data

        except BaseException as ex:
            self.logger.error(message="Exception Occurred while running inference---!! " + str(ex))
            return "Exception Occurred while running inference---!! " + str(ex)
