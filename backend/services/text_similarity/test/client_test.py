import pandas as pd

from common.logging.console_logger import ConsoleLogger
from services.text_similarity.application.ai.inference.prediction import PredictionManager
from services.text_similarity.application.ai.training.src.preprocess import Preprocess
from services.text_similarity.settings import Settings

p1 = PredictionManager(preprocess=Preprocess(), logger=ConsoleLogger(filename=Settings.LOGS_DIRECTORY))

data = pd.read_csv(Settings.TEST_DATA)
index = 55
s1 = list(data.question1.values)
s2 = list(data.question2.values)

sample_request = {
    'sentence_1': s1[index],
    'sentence_2': s2[index]
}

print("Sample Input, ", str(sample_request))
output = p1.run_inference(sample_request)
print(output)
