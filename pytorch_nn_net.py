# -*- coding: utf-8 -*-

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

LEARNING_RATE = 1e-4

# Input and Output Data ( Here we choose some random input and output to train with)
X = Variable(torch.randn(NO_OF_EXAMPLES, IN_SIZE).type(dtype))
Y = Variable(torch.randn(NO_OF_EXAMPLES, OUT_SIZE).type(dtype), requires_grad=False)

# For Graphs
loss_vals = []
line_graph = pygal.Line()
line_graph.title = 'Loss'

model = torch.nn.Sequential(
    torch.nn.Linear(IN_SIZE, HIDDEN_SIZE),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_SIZE, OUT_SIZE)
)

loss_func = torch.nn.MSELoss(size_average=False)
for episode in range(500):

    # Calculate Y_predict
    Y_pred = model(X)
    loss = loss_func(Y_pred, Y)
    loss_vals.append(loss.data[0])
    print("Episode : {} \t Loss = {}".format(episode, loss))

    # Zero the Gradient
    model.zero_grad()

    # Compute the gradients with respect to the Loss
    loss.backward()

    for parameters in model.parameters():
        parameters.data -= LEARNING_RATE * parameters.grad.data

line_graph.add('Neural Net Loss', loss_vals)
line_graph.render_to_file('./pytorch_autograd_loss.svg')
