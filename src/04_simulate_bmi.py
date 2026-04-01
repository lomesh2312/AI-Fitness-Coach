import pandas as pd
import numpy as np
import os

# Paths
HEART_DATA_PATH = "/Users/luckysonkeshriya/Desktop/pytorch/AI_Fitness_Coach/data/heart.csv"
NEW_HEART_DATA_PATH = "/Users/luckysonkeshriya/Desktop/pytorch/AI_Fitness_Coach/data/heart_with_bmi.csv"

def simulate_bmi():
    print("--- Simulating BMI for Heart Dataset ---")
    if not os.path.exists(HEART_DATA_PATH):
        print(f"Error: {HEART_DATA_PATH} not found.")
        return

    df = pd.read_csv(HEART_DATA_PATH)
    
    # Simple simulation: BMI usually correlates with Age and Blood Pressure
    # Normal BMI range: 18.5 - 40
    # Mean around 27 with some noise
    np.random.seed(42)
    base_bmi = 22 + (df['Age'] * 0.1) + (df['RestingBP'] * 0.02)
    noise = np.random.normal(0, 3, size=len(df))
    df['BMI'] = (base_bmi + noise).clip(15, 50) # Keep within realistic bounds
    
    print(f"Sample BMI values: {df['BMI'].head().tolist()}")
    
    df.to_csv(NEW_HEART_DATA_PATH, index=False)
    print(f"Saved new dataset to {NEW_HEART_DATA_PATH}")

if __name__ == "__main__":
    simulate_bmi()
