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
    output_dim = 1
    # dropout
    dropout = 0.3
    input_dim = 768
    hidden_dim = 56

    # max length for embeddings
    max_len = 256

    # bert no decay layers
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    seed_value = 42
    test_size = 0.2

    # weights path
    WEIGHTS_PATH = APPLICATION_PATH + "ai\\weights\\weights\\text_similarity_model.bin"
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    EPOCHS = 10
    RANDOM_STATE = 42
    TRAIN_NUM_WORKERS = 4
    VAL_NUM_WORKERS = 2
    patience = 4
    mode = "max"
    TARGETS_DEFAULT_KEY = -1


