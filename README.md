## Repo info
Repo goes through
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial for
PyTorch, and then replicates it in PyTorch XLA.

While doing this implementation, I decided to simplify the model, and just do a
deep neural network. I did this by substituting the RNN by a sequential
neural network, and changing so each name is now represented as a frequency
map for each letter. This is not the best translation, but will make due for
a quick example.

For the TPU example, the notebooks was translated to a main executable for TPU
best translation. Graphs are added to the directory for reference.

This has caused the model to be a lot more volatile between
runs. I am doing this case as a basic example, and I am not diving deeper, but
the behavior I saw definitely suggests overfitting.

Data pulled from `https://download.pytorch.org/tutorial/data.zip`

## PytorchXLA info
To do this migration, I used https://github.com/pytorch/xla/tree/8b45e5993bc55dec0303ca7c3a84f59326cab181?tab=readme-ov-file#getting-started.
This means that we are using the multiprocessing instructions. This has required
some minor modifications to avoid issues with multiple files being open.

Furthermore, rather than a jupiter notebook, I am relying in `main.py`.
