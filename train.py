import json
import numpy as np
import random

from utils import tokenize, bag_of_words, stem
from model import Nnt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open("intents.json", "r") as fl:
    intents = json.load(fl)

#print(intents)

all_words = []
tags = []
xy = []
for inten in intents["intents"]:
    tag = inten["tag"]
    tags.append(tag)
    for pattern in inten["patterns"]:
        wrd = tokenize(pattern)
        all_words.extend(wrd)
        xy.append((wrd, tag))

# ignore some symbols
ignore_sym = ['?', '.', '!', ',', "'", '-']
all_words = [stem(wrd) for wrd in all_words if wrd not in ignore_sym]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#print(len(xy), "patterns")
#print(len(tags), "tags:", tags)
#print(len(all_words), "unique words:", all_words)


# Creating data-set
X_train = []
Y_train = []
for (ptrn_sent, tag) in xy:
    bag = bag_of_words(ptrn_sent, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


# Hyper-parameters 
num_epochs = 800
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    #get the i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    #get lenght
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)




model = Nnt(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words
        labels = labels.to(dtype=torch.long)

        outputs = model(words)
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
