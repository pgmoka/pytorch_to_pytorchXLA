Repo goes through
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial for
PyTorch, and then replicates it in PyTorch XLA.

While doing this implementation, I decided to simplify the model, and just do a
deep neural network. I did this by substituting the RNN by a sequential
neural network, and changing so each name is now represented as a frequency
map for each letter. This is not the best translation, but will make due for
a quick example.

This has caused the model to be a lot more volatile between
runs. I am doing this case as a basic example, and I am not diving deeper, but
the behavior I saw definitely suggests overfitting.

Data pulled from `https://download.pytorch.org/tutorial/data.zip`
