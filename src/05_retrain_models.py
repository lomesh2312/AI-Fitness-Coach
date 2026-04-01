import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from models.fitness_model import FitnessModel
from models.disease_model import DiseaseModel

# Paths
DATA_DIR = "/Users/luckysonkeshriya/Desktop/pytorch/AI_Fitness_Coach/data"
CHECKPOINT_DIR = "/Users/luckysonkeshriya/Desktop/pytorch/AI_Fitness_Coach/backend/models/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train_heart_model():
    print("\n--- Training Heart Risk Model (Simplified) ---")
    data_path = os.path.join(DATA_DIR, "heart_with_bmi.csv")
    df = pd.read_csv(data_path)

    # Feature Selection (Simplified)
    # Age, Sex (0/1), BMI, RestingBP, Cholesterol
    # Mapping Sex: M/F to 1/0
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"M": 1, "F": 0})

    feature_cols = ["Age", "Sex", "BMI", "RestingBP", "Cholesterol"]
    X = df[feature_cols]
    y = df["HeartDisease"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, "disease_scaler.pkl"))

    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Model (In: 5 features)
    model = DiseaseModel(input_size=5)
    criterion = nn.BCELoss()  # Use BCELoss since Sigmoid is in forward
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train
    for epoch in range(100):
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        preds = (model(X_test_t) > 0.5).float()
        acc = accuracy_score(y_test_t, preds)
        cm = confusion_matrix(y_test_t, preds)
        print(f"Accuracy: {acc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "disease_model.pth"))
    print("Heart Model Saved.")


def train_fitness_model():
    print("\n--- Training Fitness Model (Confidence Supported) ---")
    data_path = os.path.join(DATA_DIR, "processed/processed_obesity.csv")
    if not os.path.exists(data_path):
        print("Run preprocessing (01_data_preprocessing.py) first.")
        return

    df = pd.read_csv(data_path)
    feature_cols = ["BMI", "Age", "Gender", "ActivityLevel"]
    X = df[feature_cols]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # We don't needs to rescale if processed_obesity.csv is already scaled,
    # but the training script 01_data_preprocessing.py scales it.
    # To be consistent with main.py, we should save the scaler used there.
    # However, let's just ensure we have a scaler.pkl for inference.

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.long)

    model = FitnessModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(y_test_t, preds)
        print(f"Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "fitness_model.pth"))
    print("Fitness Model Saved.")


if __name__ == "__main__":
    train_heart_model()
    train_fitness_model()
