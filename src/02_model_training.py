import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os

PROCESSED_DATA_DIR = "/Users/luckysonkeshriya/Desktop/pytorch/AI_Fitness_Coach/data/processed"
# Save directly to backend checkpoints so main.py can find them
MODELS_DIR = "/Users/luckysonkeshriya/Desktop/pytorch/AI_Fitness_Coach/backend/models/checkpoints"
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------
# MODEL 1: FITNESS
# -----------------
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# ... (Previous imports)

# -----------------
# MODEL 1: FITNESS
# -----------------
class FitnessModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, num_classes=3):
        super(FitnessModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Added for Step 2
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

def train_fitness_model():
    print("--- Training Fitness Model (Quality Update) ---")
    data_path = os.path.join(PROCESSED_DATA_DIR, "processed_obesity.csv")
    if not os.path.exists(data_path):
        X = torch.rand(100, 4); y = torch.randint(0, 3, (100,))
    else:
        df = pd.read_csv(data_path)
        feature_cols = ['BMI', 'Age', 'Gender', 'ActivityLevel']
        X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        y = torch.tensor(df['Target'].values, dtype=torch.long)
        
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = FitnessModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(50):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward(); optimizer.step()
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X), dim=1)
        acc = accuracy_score(y, preds)
        print(f"Fitness Model Accuracy: {acc*100:.2f}%")
        print("Confusion Matrix:\n", confusion_matrix(y.numpy(), preds.numpy()))

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "fitness_model.pth"))

# -----------------
# MODEL 2: DISEASE
# -----------------
class DiseaseModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=16): # Changed to 5 for Step 1
        super(DiseaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Added for Step 2
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def train_disease_model():
    print("--- Training Disease Risk Model (Industrial Overhaul) ---")
    data_path = os.path.join(PROCESSED_DATA_DIR, "processed_heart.csv")
    if not os.path.exists(data_path):
        X = torch.rand(100, 5); y = torch.randint(0, 2, (100,)).float().unsqueeze(1)
    else:
        df = pd.read_csv(data_path)
        # New 5-feature set
        feature_cols = ['Age', 'Sex', 'BMI', 'RestingBP', 'Cholesterol']
        X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        y = torch.tensor(df['HeartDisease'].values, dtype=torch.float32).unsqueeze(1)
             
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = DiseaseModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(50):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward(); optimizer.step()
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        probs = model(X)
        preds = (probs > 0.5).float()
        acc = accuracy_score(y, preds)
        print(f"Disease Model Accuracy: {acc*100:.2f}%")
        print("Confusion Matrix:\n", confusion_matrix(y.numpy(), preds.numpy()))

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "disease_model.pth"))

if __name__ == "__main__":
    train_fitness_model()
    train_disease_model()
    print("All Industrial Models Retrained Successfully!")
