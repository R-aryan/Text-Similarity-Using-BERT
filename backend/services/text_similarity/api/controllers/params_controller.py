from flask import request
from injector import inject

from backend.services.text_similarity.api.controllers.controller import Controller
from common.logging.console_logger import ConsoleLogger
from services.text_similarity.application.ai.inference.prediction import PredictionManager


class ParamsController(Controller):
    @inject
    def __init__(self, logger: ConsoleLogger, prediction: PredictionManager):
        self.logger = logger
        self.predict = prediction

    def post(self):
        try:
            req_json = request.get_json()
            result = self.predict.run_inference(req_json)
            self.predict.logger.info('Request processed successfully--!!')
            return self.response_ok(result)
        except BaseException as ex:
            self.logger.error(self.map_response('Error Occurred-- ' + str(ex)))
            return self.response_error(str(ex))

    def get(self):
        return {'response': 'This is an API endpoint for text Similarity GET Request---!!'}
