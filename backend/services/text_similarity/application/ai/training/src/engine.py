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

    def train_fn(self, data_loader, model, optimizer, device, schedular):
        print("Starting training...\n")
        # Reset the total loss for this epoch.
        total_loss, batch_loss, batch_counts = 0, 0, 0
        t0_epoch, t0_batch = time.time(), time.time()
        model.train()
        for step, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch_counts += 1
            b_input_ids = data['input_ids']
            b_attn_mask = data['attention_mask']
            b_labels = data['targets']
            b_token_type_ids = data['token_type_ids']

            # moving tensors to device
            b_input_ids = b_input_ids.to(device)
            b_attn_mask = b_attn_mask.to(device)
            b_labels = b_labels.to(device)
            b_token_type_ids = b_token_type_ids.to(device)

            # optimizer.zero_grad()

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            logits = model(
                ids=b_input_ids,
                mask=b_attn_mask,
                token_type_ids=b_token_type_ids
            )

            loss = self.loss_fn(logits, b_labels.float())
            batch_loss += loss.item()
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc
            optimizer.step()
            # Update the learning rate
            schedular.step()

            if step % 500 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = self.format_time(time.time() - t0_epoch)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(data_loader), elapsed))

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_loss / len(data_loader)
        # Measure how long this epoch took.
        training_time = self.format_time(time.time() - t0_epoch)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round(elapsed))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))
