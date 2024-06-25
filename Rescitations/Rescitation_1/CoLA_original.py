import torch
import torch.nn as nn

from pandas import read_csv
from random import shuffle

hidden_size = 500
num_classes = 2
num_epochs = 7
batch_size = 100
learning_rate = 0.001


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = read_csv('LLMs_in_data_management/Rescitations/Rescitation_1/CoLA.tsv', sep='\t', header=0)

tokens = {}
number_grammatical = 0
number_nongrammatical = 0

current_addr = 0
hot_one_address = {}
examples = []

for row in df.iterrows():
    gramatical_p = row[1].values[1]
    sentence = row[1].values[3]

    examples.append([sentence, gramatical_p])

    for token in sentence.split(' '):
        count = 1
        if token in tokens:
            count = tokens[token] + 1
        else:
            hot_one_address[token] = current_addr
            current_addr += 1

        tokens[token] = count

    if gramatical_p:
        number_grammatical += 1
        #print(f"{sentence}")
    else:
        number_nongrammatical += 1
        #print(f"*{sentence}")

shuffle(examples)

test_set = []
training_set = []

for i in range(len(examples)):
    if i < 100:
        test_set.append(examples[i])
    else:
        training_set.append(examples[i])

num_tokens = len(tokens)

# Device configuration

batch_size = 100
vocab = torch.diag(torch.ones(num_tokens,dtype=torch.float32))

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

input_size = num_tokens


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()  # remember, this does its own softmax (this should actually be BCELoss)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(silent=False):
    for epoch in range(num_epochs):
        shuffle(training_set)

        for bnum in range(80):
            batch = torch.zeros([100,num_tokens], dtype=torch.float32)
            labels = torch.zeros([100], dtype=torch.int64)
            for i in range(0,100):
                for word in training_set[bnum*100 + i][0].split(' '):
                    batch[i][hot_one_address[word]] = 1
                labels[i] = training_set[bnum*100 + i][1]

            labels.to(device)
            batch.to(device)

        # Forward pass and loss calculation
            outputs = model(batch)
            loss = criterion(outputs,labels)

        # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if not silent:
                print(f'Epoch [{epoch + 1}/{num_epochs}] Batch[{bnum+1}/80], Loss: {loss.item():.4f}')

def evaluate():
    n_correct = 0
    n_samples = len(test_set)
    with torch.no_grad():
        batch = torch.zeros([100, num_tokens], dtype=torch.float32)
        labels = torch.zeros([100], dtype=torch.int64)
        for i in range(0, 100):
            for word in test_set[i][0].split(' '):
                batch[i][hot_one_address[word]] = 1
            labels[i] = test_set[i][1]
        batch.to(device)
        labels.to(device)

        outputs = model(batch)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

        acc = n_correct / n_samples
        print(f'Accuracy of the network on the {n_samples} test images: {100 * acc:.4f} %')

evaluate()
train(silent=True)
evaluate()
