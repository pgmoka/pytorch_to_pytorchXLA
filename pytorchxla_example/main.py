import torch
from torch.utils.data import Dataset
import torch_xla as xla
import torch_xla.core.xla_model as xm
import matplotlib
import matplotlib.pyplot as plt
import time
import numpy as np
import string
import unicodedata
import os
import psutil
import faulthandler

import dnn as dnn_helper
import namesdataset as nd

faulthandler.enable()

if __name__ == '__main__':
  print('--- Start main')

  print('--- Prepare and load data')

  allowed_characters = string.ascii_letters + " .,;'"
  n_letters = len(allowed_characters)

  # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
  def unicodeToAscii(s):
      return ''.join(
          c for c in unicodedata.normalize('NFD', s)
          if unicodedata.category(c) != 'Mn'
          and c in allowed_characters
      )

  alldata = nd.NamesDataset("../data/names")
  print(f"loaded {len(alldata)} items of data")
  print(f"example = {alldata[0]}")

  train_set, test_set = torch.utils.data.random_split(alldata, [.85, .15], generator=torch.Generator().manual_seed(2024))

  print(f"train examples = {len(train_set)} (type: {type(train_set)}), validation examples = {len(test_set)}")

  print('--- Prepare model')

  n_hidden = 128
  dnn = dnn_helper.DeepANN(n_letters, n_hidden, len(alldata.labels_uniq))
  print(dnn)

  print('- Pre-training sample output:')
  input = nd.lineToTensor('Albert')
  output = dnn(input) #this is equivalent to ``output = dnn.forward(input)``
  print(output)
  print(dnn_helper.label_from_output(output, alldata.labels_uniq))

  print('--- Training model')

  process = psutil.Process(os.getpid())
  print(f" Files open before training: {process.open_files()}")

  start = time.time()

  xla.launch(dnn_helper.train, args=(dnn, train_set, 5, 0.15, 5, 3))
  # all_losses = train(dnn, train_set, n_epoch=27, learning_rate=0.15, report_every=5)
  end = time.time()
  print(f"training took {end-start}s")

  print('--- Model evaluation')

  print('Post-training sample output:')

  input = nd.lineToTensor('Albert')
  output = dnn(input) #this is equivalent to ``output = dnn.forward(input)``
  print(output)
  print(dnn_helper.label_from_output(output, alldata.labels_uniq))

  # all_losses = np.array(all_losses)
  # all_losses

  # plt.figure()
  # plt.plot(all_losses)
  # plt.show()

  # dnn_helper.evaluate(dnn, test_set, classes=alldata.labels_uniq)
