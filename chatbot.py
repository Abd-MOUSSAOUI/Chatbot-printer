import random
import json
import torch

from model import Nnt
from utils import bag_of_words, tokenize


with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)