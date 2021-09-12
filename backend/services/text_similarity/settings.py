import os

import torch
from transformers import AutoTokenizer


class Settings:
    PROJ_NAME = 'Text-Similarity-Using-BERT'
    root_path = os.getcwd().split(PROJ_NAME)[0] + PROJ_NAME + "\\"
    APPLICATION_PATH = root_path + "backend\\services\\text_similarity\\application\\"
    # setting up logs path
    LOGS_DIRECTORY = root_path + "backend\\services\\text_similarity\\logs\\logs.txt"

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
