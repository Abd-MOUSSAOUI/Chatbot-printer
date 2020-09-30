import random
import json
import torch
import re
import datetime
import os


from model import Nnt
from utils import bag_of_words, tokenize


with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = Nnt(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

counter = 0
bot_name = "Printer-Bot"
print("\t --> Bonjour, je suis un Chatbot qui organise l'impression des document. \n\tJe peux vous aidez à imprimer un documents \n\tPour cela, il suffit juste de me dire le nom du documents et combien de pages il cotient.. \n\tPar ex: je veux imprimer Doc1 qui cotient 2345 pages\n\n\tTape quit or bye pour sortir")

report = open("report.txt", "a")
agenda = open("agenda.txt", "a")


while True:
    sentence = input("You: ")
    if (sentence == "quit" or sentence == "bye"):
        print("Printer-Bot: Au revoir :)")
        break
    date = datetime.datetime.now()
    sent = tokenize(sentence)
    X = bag_of_words(sent, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() >= 0.95:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        tag = "other"
        print(f"{bot_name}: Sorry ! I don't understand... tape 'help' to see what i can do")
    report.write("\n" + sentence +"\t" + str(date)+"\t" + tag + "\n")
    if tag == 'print':
        time = nbr_pages = 0
        for word in sent:
            if word.isdigit():
                nbr_pages = int(word)
                time = date.time().hour*3600 + date.time().minute*60 + date.time().second
        if time >= counter:
            counter = time + nbr_pages
        else:
            counter = counter + nbr_pages
    # trosform seconds to time
    # str(datetime.timedelta(seconds=666))

report.close()
agenda.close()