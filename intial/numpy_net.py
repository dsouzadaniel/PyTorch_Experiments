# An implementation of a simple 2-layer Neural Net using just numpy and random inputs and outputs

# Libraries
import numpy as np
import pygal


# Dimensions
IN_SIZE = 1000
HIDDEN_SIZE = 100
OUT_SIZE = 10
NO_OF_EXAMPLES = 64

LEARNING_RATE = 1e-6

# Input and Output Data ( Here we choose some random input and output to train with)
X = np.random.randn(NO_OF_EXAMPLES, IN_SIZE)
Y = np.random.randn(NO_OF_EXAMPLES, OUT_SIZE)

# Random Weights
W1 = np.random.randn(IN_SIZE, HIDDEN_SIZE)
W2 = np.random.randn(HIDDEN_SIZE, OUT_SIZE)

# For Graphs
loss_vals = []
line_graph = pygal.Line()
line_graph.title = 'Loss'

for episode in range(500):
    H1 = X.dot(W1)
    H1r = np.maximum(H1, 0)
    Y_pred = H1.dot(W2)

    loss = np.square(Y_pred - Y).sum()
    loss_vals.append(loss)
    print("Episode : {} \t Loss = {}".format(episode, loss))
    # print("Loss :",loss)

    grad_Y_pred = 2.0 * (Y_pred - Y)
    grad_W2 = H1r.T.dot(grad_Y_pred)
    grad_H1r = grad_Y_pred.dot(W2.T)
    grad_H = grad_H1r.copy()
    grad_H[H1 < 0] = 0
    grad_W1 = X.T.dot(grad_H)

    W1-= LEARNING_RATE * grad_W1
    W2-= LEARNING_RATE * grad_W2

line_graph.add('Neural Net Loss',loss_vals)
line_graph.render_to_file('./loss.svg')
