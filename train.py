import json
import numpy as np
import random

import nltk
nltk.download('punkt')

from utils import tokenize, bag_of_words, stem
from model import Nnt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open("intents.json", "r") as f:
    intents = json.load(f)

#print(intents)

all_words = []
tags = []
x_y = []
for inten in intents["intents"]:
    tag = inten["tag"]
    tags.append(tag)
    for pattern in inten["patterns"]:
        wrd = tokenize(pattern)
        all_words.extend(wrd)
        x_y.append((wrd, tag))

# ignore some symbols
ignore_sym = ['?', '.', '!', ',', "'", '-']
all_words = [stem(wrd) for wrd in all_words if wrd not in ignore_sym]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#print(len(x_y), "patterns")
#print(len(tags), "tags:", tags)
#print(len(all_words), "unique words:", all_words)


# Creating data-set
X_train = []
Y_train = []
for (ptrn_sent, tag) in x_y:
    bag = bag_of_words(ptrn_sent, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


# Parameters 
input_size = len(X_train[0])
output_size = len(tags)
batch_size = 8
hidden_size = 8
learning_rate = 0.001
epochs = 800

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
train = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)


model = Nnt(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    for (words, labels) in train:
        words = words
        labels = labels.to(dtype=torch.long)

        outputs = model(words)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

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
