

import random
import math
import torch
from examples import xor, majority, one_wire_not

learning_rate = 0.01




def dot_prod(inputs,w):
    z = w[0]  # this is the bias
    for i in range(1,4):
        z += inputs[i-1] * w[i]
    return z

def ReLU(x,derivative = False):
    if x>=0 and not derivative:
        return x
    if x>=0:
        return 1
    return 0

def step(x):
    if x <= 0:
        return 0
    elif x > 0:
        return 1

def sigmoid(x,derivative = False):
    if not derivative:
        return 1/(1 +  math.exp(-x))
    else:
        return sigmoid(x)*(1-sigmoid(x))

def tanh(x,derivative = False):
    if not derivative:
        return math.tanh(x)
    else:
        return 1 - math.tanh(x)**2

def uniform_0_1():
    return random.random()

def uniform_m1_p1():
    return (random.random() * 2) - 1

def zeros():
    return 0

def forward_and_backward(example, g, w):
    grad = [0, 0, 0, 0]
    dot = dot_prod(example[0],w)
    answer = g(dot)
    err =  example[1] - answer
    loss = .5 * (err **2)
    derivative = g(dot, derivative=True)
    grad[0] = -err * derivative * 1
    for i in range(1, 4):
        grad[i] = -err * derivative * example[0][i - 1]
    return answer,err,loss,grad

def update(gradient, w):
    for i in range(0,4):
        w[i] = w[i] - learning_rate * gradient[i]

def train(g, examples, initial_weight_distribution, epochs=100, print_weights=False):
    w = [initial_weight_distribution() for i in range(4)]
    for epoch in range(epochs):
        total_loss = 0
        for example in examples:
            answer,err,loss,grad = forward_and_backward(example, g, w)
            total_loss += loss
            if print_weights:
                print(f"weights: {w[0]:.4f} {w[1]:.4f} {w[2]:.4f} {w[3]:.4f}")
            print(f"{str(example)}:{answer:.4f}, error={err:.4f}, loss={loss:.4f}, grad={grad[0]:.4f} {grad[1]:.4f} {grad[2]:.4f} {grad[3]:.4f}")
            update(grad, w)
        print(f"average loss {total_loss/len(examples):.4f} \n")

train(ReLU, majority,uniform_0_1,epochs=100,print_weights=False)
#train(ReLU, majority,uniform_m1_p1,epochs=100,print_weights=False)
#train(ReLU, majority,zeros,epochs=100,print_weights=False)
#train(ReLU, one_wire_not,uniform_0_1,epochs=100,print_weights=False)
#train(ReLU, one_wire_not,uniform_m1_p1,epochs=100,print_weights=False)
#train(ReLU, one_wire_not,zeros,epochs=100,print_weights=False)
