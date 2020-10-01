import torch
import torch.nn as nn

class Nnt(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Nnt, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size) 
        self.lin2 = nn.Linear(hidden_size, hidden_size) 
        self.lin3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.relu(x)
        out = self.lin1(out)
        out = self.lin2(out)
        out = self.lin3(out)
        return out


    # def forward(self, x):
    #     out = self.lin1(x)
    #     out = self.relu(out)
    #     out = self.lin2(out)
    #     out = self.relu(out)
    #     out = self.lin3(out)
    #     return out