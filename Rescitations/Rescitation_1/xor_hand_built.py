
from perceptron import specific_initialization, xor_examples, train

# Initialize weights and train
weights = specific_initialization()
train(xor_examples, weights, epochs=100, learning_rate=0.01, regularization_lambda=0.01)

#from examples import xor_hand_built
#import perceptron as per
#
#constant_parameters = [
#    [1,1,-0,5],
#    [1,-1,-0,5],
#    [-1,1,-0,5],
#    [1,-1,1,-1,5]]
#
#
#def forward_and_backward_xor(example, weights):
#    x, y = example
#    hidden_layer = [0] * 3
#    for i in range(3):
#        hidden_layer[i] = per.ReLU(per.dot_prod(x, weights[i]))
#    
#    output = per.step(per.dot_prod(hidden_layer, weights[3]))
#    
#    err = y - output
#    loss = 0.5 * (err ** 2)
#
#    return output, err, loss
#
#def train(examples, weights, epochs = 10):
#    for epoch in range(epochs):
#        total_loss = 0
#        for example in examples:
#            output, err, loss = forward_and_backward_xor(example, weights)
#            total_loss += loss
#            print(f"Input: {example[0]}, Output: {output}, Target: {example[1]}, Error: {err}, Loss: {loss}")
#        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(examples)}\n")
#
#
#train(xor_hand_built, constant_parameters)