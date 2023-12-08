from NueralNetworks.nueral_network import NueralNetwork
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# prep data
test_directory = './NueralNetworks/bank-note/test.csv'
train_directory = './NueralNetworks/bank-note/train.csv'
test_data = pd.read_csv(test_directory, header=None)
train_data = pd.read_csv(train_directory, header=None)
test_data = test_data.to_numpy()
for t in test_data:
    if t[-1] == 0:
        t[-1] = -1
train_data = train_data.to_numpy()
for t in train_data:
    if t[-1] == 0:
        t[-1] = -1
ones = np.ones((len(test_data), 1))
test_data = np.column_stack((ones, test_data))
ones = np.ones((len(train_data), 1))
train_data = np.column_stack((ones, train_data))

# part a: forward test
nn = NueralNetwork(3)
nn.add_layer(-1, 3)
nn.add_layer(-1, 3)
nn.add_layer(-1, 1)
# weights = [
#     # node z_1^1
#     (1, 1, [[0, -1, 0], [1, -2, 0], [2, -3, 0]]),
#     # node z_2^1
#     (1, 2, [[0, 1, 0], [1, 2, 0], [2, 3, 0]]),
#     # node z_1^2
#     (2, 1, [[0, -1, 0], [1, -2, 0], [2, -3, 0]]),
#     # node z_2^2
#     (2, 2, [[0, 1, 0], [1, 2, 0], [2, 3, 0]]),
#     # node y
#     (3, 0, [[0, -1, 0], [1, 2, 0], [2, -1.5, 0]])
# ]
# nn.add_weights(weights)
nn.add_weights('gaussian')
input = [[1], [1], [1]]
output = nn.forward(input)[0]
print(output)

# # backward test
ystar = 1
gradient = nn.backwards(ystar, output)
print(gradient)

# part b: general training
# 5 nodes per layer
nn = NueralNetwork(4)
nn.add_layer(-1, 5)
nn.add_layer(-1, 5)
nn.add_layer(-1, 1)
nn.add_weights('zero', None)
train_errors, errors = nn.train(train_data, test_data, 1e-4, 1, 1e-5)
print(errors)
plt.plot(train_errors)
plt.show()
# 10 nodes per layer
nn = NueralNetwork(4)
nn.add_layer(-1, 10)
nn.add_layer(-1, 10)
nn.add_layer(-1, 1)
nn.add_weights('zero', None)
train_errors, errors = nn.train(train_data, test_data, 1e-4, 1, 1e-5)
print(errors)
plt.plot(train_errors)
plt.show()
# 25 nodes per layer
nn = NueralNetwork(4)
nn.add_layer(-1, 25)
nn.add_layer(-1, 25)
nn.add_layer(-1, 1)
nn.add_weights('zero', None)
train_errors, errors = nn.train(train_data, test_data, 1e-3, 1, 1e-5)
print(errors)
plt.plot(train_errors)
plt.show()
# 50 nodes per layer
nn = NueralNetwork(4)
nn.add_layer(-1, 50)
nn.add_layer(-1, 50)
nn.add_layer(-1, 1)
nn.add_weights('zero', None)
train_errors, errors = nn.train(train_data, test_data, 1e-3, 1, 1e-5)
print(errors)
plt.plot(train_errors)
plt.show()
# 50 nodes per layer
nn = NueralNetwork(4)
nn.add_layer(-1, 100)
nn.add_layer(-1, 100)
nn.add_layer(-1, 1)
nn.add_weights('zero', None)
train_errors, errors = nn.train(train_data, test_data, 1e-3, 1, 1e-5)
print(errors)
plt.plot(train_errors)
plt.show()

# part e PyTorch
# need to make a DataSet
class ArrayDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (self.data[idx][0:-1], np.array([self.data[idx][-1]]))
        # sample = {'x': self.data[idx][0:-1], 'y': self.data[idx][-1]}
        return sample

BATCH_SIZE = 1
training_data = ArrayDataset(train_data)
testing_data = ArrayDataset(test_data)
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print('using: ', device)

# define the model (depth from {3, 5, 9} width from {5, 10, 25, 50, 100})
class NueralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NueralNetwork().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# training function
def train(dataloader, model, loss_fn, optimizer):
    # size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(torch.float32)
        y = y.to(torch.float32)
        X, y = X.to(device), y.to(device)

        # prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1)*len(X)
        #     print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# test function
def test(dataloader, model, loss_fn, msg):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(torch.float32)
            y = y.to(torch.float32)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.sign() == y.sign()).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"{msg} Error: \n Inaccuracy: {(100 - 100*correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n---------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn, 'test')
    test(train_dataloader, model, loss_fn, 'train')
print("Done!")