import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


def calculate_bmi(weight_lbs, height_inches):
    """Calculates BMI using imperial units."""
    if height_inches == 0:
        return 0
    return (weight_lbs / (height_inches**2)) * 703


def preprocess_fitness_data(csv_path):
    """
    Preprocesses the BodyFat dataset for the Fitness Model.
    Target: 0=Poor, 1=Average, 2=Fit
    Features: BMI, Age, Gender (Male by default), ActivityLevel (Simulated)
    """
    df = pd.read_csv(csv_path)

    # 1. Calculate BMI
    df["BMI"] = df.apply(
        lambda row: calculate_bmi(row["Weight"], row["Height"]), axis=1
    )

    # 2. Map Target (Fitness Level) based on BodyFat percentage
    # Scientific thresholds for men (the dataset is mostly men):
    # Fit: < 18%, Average: 18-24%, Poor: 25%+
    def map_fitness(bf):
        if bf < 18:
            return 2  # Fit
        elif bf < 25:
            return 1  # Average
        else:
            return 0  # Poor

    df["target"] = df["BodyFat"].apply(map_fitness)

    # 3. Add Gender and ActivityLevel (Simulated for training since missing)
    # We use a Seed for reproducibility
    np.random.seed(42)
    df["Gender"] = 1  # 1 for Male
    df["ActivityLevel"] = np.random.randint(
        0, 3, size=len(df)
    )  # 0: Low, 1: Med, 2: High

    # 4. Select Final Features
    features = ["BMI", "Age", "Gender", "ActivityLevel"]
    X = df[features]
    y = df["target"]

    return X, y


def preprocess_disease_data(csv_path):
    """
    Preprocesses the Heart Disease dataset for the Disease Model.
    Target: 0=No Disease, 1=Disease
    Features: Age, Cholesterol, RestingBP, MaxHR
    """
    df = pd.read_csv(csv_path)

    # 1. Select relevant features
    features = ["Age", "Cholesterol", "RestingBP", "MaxHR"]

    # 2. Handle 0 values in Cholesterol (common in this dataset)
    # Replace 0 with the median of non-zero values
    chol_median = df[df["Cholesterol"] > 0]["Cholesterol"].median()
    df["Cholesterol"] = df["Cholesterol"].replace(0, chol_median)

    X = df[features]
    y = df["HeartDisease"]

    return X, y


def get_scalers(fitness_X, disease_X):
    """Fits and returns scalers to be reused in the API."""
    fitness_scaler = StandardScaler()
    fitness_scaler.fit(fitness_X)

    disease_scaler = StandardScaler()
    disease_scaler.fit(disease_X)

    return fitness_scaler, disease_scaler
