# An implementation of a simple 2-layer Neural Net using just PyTorch Tensors ( no autograd)

# Libraries
import pygal

import torch

dtype = torch.FloatTensor

# Dimensions
IN_SIZE = 1000
HIDDEN_SIZE = 100
OUT_SIZE = 10
NO_OF_EXAMPLES = 64

LEARNING_RATE = 1e-6

# Input and Output Data ( Here we choose some random input and output to train with)
X = torch.randn(NO_OF_EXAMPLES, IN_SIZE).type(dtype)
Y = torch.randn(NO_OF_EXAMPLES, OUT_SIZE).type(dtype)

# Random Weights
W1 = torch.randn(IN_SIZE, HIDDEN_SIZE).type(dtype)
W2 = torch.randn(HIDDEN_SIZE, OUT_SIZE).type(dtype)

# For Graphs
loss_vals = []
line_graph = pygal.Line()
line_graph.title = 'Loss'

for episode in range(500):
    H1 = X.mm(W1)
    H1r = H1.clamp(min=0)
    Y_pred = H1.mm(W2)

    loss = (Y_pred - Y).pow(2).sum()
    loss_vals.append(loss)
    print("Episode : {} \t Loss = {}".format(episode, loss))
    # print("Loss :",loss)

    grad_Y_pred = 2.0 * (Y_pred - Y)
    grad_W2 = H1r.t().mm(grad_Y_pred)
    grad_H1r = grad_Y_pred.mm(W2.t())
    grad_H = grad_H1r.clone()
    grad_H[H1 < 0] = 0
    grad_W1 = X.t().mm(grad_H)

    W1-= LEARNING_RATE * grad_W1
    W2-= LEARNING_RATE * grad_W2

line_graph.add('Neural Net Loss',loss_vals)
line_graph.render_to_file('./pytorch_loss.svg')
