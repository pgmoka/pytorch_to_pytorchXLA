import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import time
import numpy as np

print('-- Initiate model and training declaration')

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)

        return output

def train(rnn, training_data, n_epoch = 10, n_batch_size = 64, report_every = 50, learning_rate = 0.2, criterion = nn.NLLLoss()):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"training on data set with n = {len(training_data)}")

    rnn.to(xla.device())

    for iter in range(1, n_epoch + 1):
        with xla.step():
          rnn.zero_grad() # clear the gradients

          # create some minibatches
          # we cannot use dataloaders because each of our names is a different length
          batches = list(range(len(training_data)))
          random.shuffle(batches)
          batches = np.array_split(batches, len(batches) //n_batch_size )

          for idx, batch in enumerate(batches):
              idx, batch = idx.to(xla.device()), batch.to(xla.device())
              batch_loss = 0
              for i in batch: #for each example in this batch
                  (label_tensor, text_tensor, label, text) = training_data[i]
                  output = rnn.forward(text_tensor)
                  loss = criterion(output, label_tensor)
                  batch_loss += loss

              # optimize parameters
              batch_loss.backward()
              nn.utils.clip_grad_norm_(rnn.parameters(), 3)
              xm.optimizer_step(optimizer)
              optimizer.zero_grad()

              current_loss += batch_loss.item() / len(batch)

          all_losses.append(current_loss / len(batches) )
          if iter % report_every == 0:
              print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
          current_loss = 0

    return all_losses

def evaluate(rnn, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))

    rnn.eval() #set to eval mode
    with torch.no_grad(): # do not record the gradients during eval phase
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = rnn(text_tensor)
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


print('-- Finished model and training declaration')
