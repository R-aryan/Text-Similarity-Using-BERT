import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import datetime


class Engine:
    def __init__(self):
        pass

    def loss_fn(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def set_seed(self, seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
