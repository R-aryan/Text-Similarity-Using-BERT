from flask import request
from injector import inject

from backend.services.text_similarity.api.controllers.controller import Controller
# from services.text_summarization.application.ai.inference.prediction_manager import PredictionManager
# from services..settings import Settings


class ParamsController(Controller):
    @inject
    def __init__(self):
        pass

    def post(self):
        try:
            # req_json = request.get_json()
            # result = self.predict.run_inference(req_json[Settings.DATA_KEY])
            # self.predict.logger.info('Request processed successfully--!!')
            # return self.response_ok(result)
            return {'response': 'This is an API endpoint for text Similarity---!!'}
        except BaseException as ex:
            self.predict.logger.error(self.map_response('Error Occurred-- ' + str(ex)))
            return self.response_error(str(ex))

    def get(self):
        return {'response': 'This is an API endpoint for text summarization---!!'}
