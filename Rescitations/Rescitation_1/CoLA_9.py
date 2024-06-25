import torch
import torch.nn as nn

from pandas import read_csv
from random import shuffle


hidden_size = 100
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
    def __init__(self, input_size, hidden_size,num_classes):
        super(NeuralNet, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        out = self.fc(out[:, -1, :])
        return out

input_size = num_tokens


model = NeuralNet(input_size, hidden_size, num_classes)
model.to(device)

criterion = nn.BCEWithLogitsLoss()  # remember, this does its own softmax (this should actually be BCELoss)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[i] for i in seq.split(' ')]
    return torch.tensor(idxs, dtype=torch.long)

def pad_sequences(batch):
    max_len = max(len(seq) for seq in batch)
    padded_batch = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded_batch[i, :len(seq)] = torch.tensor(seq, dtype=torch.long).clone().detach()
    return padded_batch
    
def train(silent=False):
    for epoch in range(num_epochs):
        shuffle(training_set)

        for bnum in range(80):
            sentence = [prepare_sequence(training_set[i][0],hot_one_address) for i in range(bnum * batch_size, (bnum + 1) * batch_size)]
            labels = torch.tensor([training_set[i][1] for i in range(bnum * batch_size, (bnum + 1) * batch_size)], dtype=torch.float32).unsqueeze(1)
            batch = pad_sequences(sentence)

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
        sentence = [prepare_sequence(test_set[i][0], hot_one_address) for i in range(n_samples)]
        labels = torch.tensor([test_set[i][1] for i in range(n_samples)], dtype=torch.float32).unsqueeze(1) # = torch.zeros([100,1], dtype=torch.float32)
        batch = pad_sequences(sentence)

        batch.to(device)
        labels.to(device)

        outputs = model(batch)
        predicted = torch.round(torch.sigmoid(outputs))#.squeeze() #_, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

        acc = n_correct / n_samples
        print(f'Accuracy of the network on the {n_samples} test images: {100 * acc:.4f} %')

evaluate()
train(silent=True)
evaluate()
