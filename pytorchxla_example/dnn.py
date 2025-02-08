import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import numpy as np
import random
import os
import psutil

print('-- Initiate model and training declaration')

class DeepANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_deep_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        # x = self.flatten(line_tensor)
        output = self.linear_deep_stack(line_tensor)
        output = self.softmax(output)

        return output

def train(model, training_data, n_epoch = 10, n_batch_size = 64, report_every = 50, learning_rate = 0.2, criterion = nn.NLLLoss()):
  """
  Learn on a batch of training_data for a specified number of iterations and reporting thresholds
  """
  current_loss = 0
  all_losses = []
  model.train()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  start = time.time()

  # Move model to TPU
  model.to(xla.device())
  for iter in range(1, n_epoch + 1):
    # Use XLA step
    with xla.step():
      model.zero_grad()

      batches = list(range(len(training_data)))
      random.shuffle(batches)
      batches = np.array_split(batches, len(batches) //n_batch_size )

      for idx, batch in enumerate(batches):
        batch_loss = 0
        for i in batch:
          (label_tensor, text_tensor, label, text) = training_data[i]
          # Move training data to XLA
          text_tensor, label_tensor = text_tensor.to(xla.device()), label_tensor.to(xla.device())
          output = model.forward(text_tensor)
          loss = criterion(output, label_tensor)
          batch_loss += loss
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        optimizer.zero_grad()

        current_loss += batch_loss.item() / len(batch)
        xm.mark_step()
        xm.wait_device_ops()

      all_losses.append(current_loss / len(batches) )
      if iter % report_every == 0:
        print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
      current_loss = 0

  return all_losses

def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i

def evaluate(model, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))

    model.eval() #set to eval mode
    with torch.no_grad(): # do not record the gradients during eval phase
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = model(text_tensor.to('xla'))
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(len(classes)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy()) #numpy uses cpu here so we need to use a cpu version
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()
    plt.savefig('evaluation.png')


print('-- Finished model and training declaration')

# if __name__ == '__main__':
#   xla.launch(train, args=((dnn, train_set, 27, 0.15, 5, 10)))
