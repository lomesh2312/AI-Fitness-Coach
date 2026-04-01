import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Define paths
DATA_DIR = "/Users/luckysonkeshriya/Desktop/pytorch/AI_Fitness_Coach/data"
OBESITY_DATA_PATH = os.path.join(DATA_DIR, "bodyfat.csv")
HEART_DATA_PATH = os.path.join(DATA_DIR, "heart.csv")

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_obesity_dataset():
    print("--- Preprocessing Body Fat Dataset ---")
    if not os.path.exists(OBESITY_DATA_PATH):
        print(f"File not found: {OBESITY_DATA_PATH}. Please check the folder.")
        return
    
    # Load dataset
    df = pd.read_csv(OBESITY_DATA_PATH)
    df = df.dropna()
    
    # BMI calculation (Weight is in lbs, Height in inches)
    # BMI = (Weight_lbs / Height_inches^2) * 703
    df['BMI'] = (df['Weight'] / (df['Height'] ** 2)) * 703
    
    # Deriving Fitness Level from BodyFat percentage
    # Standard Men ranges: < 18% (Fit), 18-24% (Average), 25%+ (Poor)
    def categorize_fitness(bf):
        if bf < 18:
            return 'Fit'
        elif bf < 25:
            return 'Average'
        else:
            return 'Poor'
            
    df['Fitness_Level'] = df['BodyFat'].apply(categorize_fitness)
    
    # Encode target: Fit=2, Average=1, Poor=0
    target_map = {'Poor': 0, 'Average': 1, 'Fit': 2}
    df['Target'] = df['Fitness_Level'].map(target_map)
    
    # Selection of features (Using only what the user can input later)
    # Adding a dummy 'Gender' column (Male=1) as most of this data is male
    df['Gender'] = 1
    # Adding a dummy 'ActivityLevel' (Average=1) 
    df['ActivityLevel'] = 1
    
    # Features: BMI, Age, Gender, ActivityLevel
    # This matches the model we defined in main.py and training script
    feature_cols = ['BMI', 'Age', 'Gender', 'ActivityLevel']
    
    # Normalize numeric data
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save scaler
    scaler_path = os.path.join("/Users/luckysonkeshriya/Desktop/pytorch/AI_Fitness_Coach/backend/models/checkpoints", "fitness_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved Fitness Scaler to {scaler_path}")
    
    # Save processed dataframe
    out_path = os.path.join(PROCESSED_DIR, "processed_obesity.csv")
    final_df = df[feature_cols + ['Target']]
    final_df.to_csv(out_path, index=False)
    print(f"Saved processed dataset (for Fitness Model) to {out_path}\n")

def preprocess_heart_dataset():
    print("--- Preprocessing Heart Disease Dataset (Industrial Fix) ---")
    if not os.path.exists(HEART_DATA_PATH):
        print(f"File not found: {HEART_DATA_PATH}.")
        return
        
    df = pd.read_csv(HEART_DATA_PATH)
    df = df.dropna()
    
    # 1. Encode Sex (Male=1, Female=0)
    if 'Sex' in df.columns:
        le = LabelEncoder()
        df['Sex'] = le.fit_transform(df['Sex']) # Usually M/F -> 1/0
    else:
        df['Sex'] = 1 # Default
        
    # 2. Synthesize BMI (Since it's missing in basic heart datasets)
    # We generate a realistic BMI based on Age and a random normal distribution
    # This keeps the model architecture consistent for the user's 5-feature requirement
    np.random.seed(42)
    df['BMI'] = 22 + (df['Age'] * 0.1) + np.random.normal(0, 3, size=len(df))
    df['BMI'] = df['BMI'].clip(15, 45) # Keep in realistic human range
            
    # 3. New 5-Feature Set: Age, Sex, BMI, RestingBP, Cholesterol
    feature_cols = ['Age', 'Sex', 'BMI', 'RestingBP', 'Cholesterol']
    
    # Check if all columns exist
    for col in feature_cols:
        if col not in df.columns:
            print(f"Column {col} missing in dataset. Using defaults.")
            df[col] = df.get(col, 0)
            
    # 4. Normalize
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 5. Save Scaler
    scaler_path = os.path.join("/Users/luckysonkeshriya/Desktop/pytorch/AI_Fitness_Coach/backend/models/checkpoints", "disease_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved Industrial 5-Feature Scaler to {scaler_path}")
    
    # 6. Save Processed Data
    out_path = os.path.join(PROCESSED_DIR, "processed_heart.csv")
    final_df = df[feature_cols + ['HeartDisease']]
    final_df.to_csv(out_path, index=False)
    print(f"Saved 5-feature heart dataset to {out_path}\n")

if __name__ == "__main__":
    print("Starting data preprocessing...")
    preprocess_obesity_dataset()
    preprocess_heart_dataset()
    print("Done!")
