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

    # training data directory
    TRAIN_DATA = APPLICATION_PATH + "ai\\data\\train.csv"

    # test data directory
    TEST_DATA = APPLICATION_PATH + "ai\\data\\test.csv"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # labels
    possible_labels = {'not_duplicate': 0, 'duplicate': 1}
    # number of labels
    num_labels = 1
    # dropout
    dropout = 0.3
    input_dim = 768

    # max length for embeddings
    max_len = 256
