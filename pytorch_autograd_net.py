# An implementation of a simple 2-layer Neural Net using just PyTorch Tensors ( no autograd)

# Libraries
import pygal

import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

# Dimensions
IN_SIZE = 1000
HIDDEN_SIZE = 100
OUT_SIZE = 10
NO_OF_EXAMPLES = 64

LEARNING_RATE = 1e-6

# Input and Output Data ( Here we choose some random input and output to train with)
X = Variable(torch.randn(NO_OF_EXAMPLES, IN_SIZE).type(dtype), requires_grad=False)
Y = Variable(torch.randn(NO_OF_EXAMPLES, OUT_SIZE).type(dtype), requires_grad=False)

# Random Weights
W1 = Variable(torch.randn(IN_SIZE, HIDDEN_SIZE).type(dtype), requires_grad=True)
W2 = Variable(torch.randn(HIDDEN_SIZE, OUT_SIZE).type(dtype), requires_grad=True)

# For Graphs
loss_vals = []
line_graph = pygal.Line()
line_graph.title = 'Loss'

for episode in range(500):
    # Calculate Y_predict
    Y_pred = X.mm(W1).clamp(min=0).mm(W2)

    loss = (Y_pred - Y).pow(2).sum()
    loss_vals.append(loss.data[0])
    print("Episode : {} \t Loss = {}".format(episode, loss))

    # Compute the gradients with respect to the Loss
    loss.backward()

    # Update the Weights easily using the AutoGrad package.
    W1.data-= LEARNING_RATE * W1.grad.data
    W2.data-= LEARNING_RATE * W2.grad.data

    # Manually Zero the Gradients after each weight update
    W1.grad.data.zero_()
    W2.grad.data.zero_()


line_graph.add('Neural Net Loss',loss_vals)
line_graph.render_to_file('./pytorch_autograd_loss.svg')