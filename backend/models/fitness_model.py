import torch
import torch.nn as nn


class FitnessModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, num_classes=3):
        super(FitnessModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def get_confidence(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
