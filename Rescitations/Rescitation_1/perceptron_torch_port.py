import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import perceptron
import examples


##possible the models below
class simple_neural_net(nn.Module):
    def __init__ (self):
        super(simple_neural_net,self).__init__()
        self.linear = nn.Linear(3,1)
    def forward(self,x):
        x = self.linear(x)
        return x
    
activation_functions = {
    'perceptron.ReLU':nn.ReLU(),
    'perceptron.sigmoid':nn.Sigmoid(),
    'perceptron.tanh':nn.Tanh(),
    'perceptron.step': lambda x: torch.where(x <= 0, torch.tensor(0.0), torch.tensor(1.0))
}


def train(model, examples, epochs = 100, print_weights = False):
    for epoch in range(epochs):
        total_loss = 0
        for example in examples:
            inputs = torch.tensor(example[0], dtype=torch.float32)
            target = torch.tensor(example[1], dtype=torch.float32)

            output = model(inputs)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if print_weights:
                print(f"Weights: {model.linear.weight.data.numpy()[0]} Bias: {model.linear.bias.data.numpy()[0]}")
            
            print(f"{str(example)}: {output.item():.4f}, loss={loss.item():.4f}")
        
        print(f"Average loss {total_loss / len(examples):.4f} \n")

#learning_rate = 0.01
model = simple_neural_net()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = perceptron.learning_rate)

train(model, examples.one_wire_not, epochs=100, print_weights=False)