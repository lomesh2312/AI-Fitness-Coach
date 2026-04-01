import pandas as pd
import json
import os

# Paths
FOOD_DATA_DIR = "/Users/luckysonkeshriya/Desktop/pytorch/AI_Fitness_Coach/FINAL FOOD DATASET"
KNOWLEDGE_BASE_PATH = "/Users/luckysonkeshriya/Desktop/pytorch/AI_Fitness_Coach/backend/rag/knowledge_base.json"

def generate_knowledge_base():
    print("--- Generating Food Knowledge Base ---")
    
    # Categories to extract from CSVs
    food_facts = []
    
    # Loop through all 5 groups
    for i in range(1, 6):
        file_path = os.path.join(FOOD_DATA_DIR, f"FOOD-DATA-GROUP{i}.csv")
        if not os.path.exists(file_path):
            print(f"Skipping {file_path}, not found.")
            continue
            
        df = pd.read_csv(file_path)
        # Select important columns
        # food, Caloric Value, Fat, Protein, Carbohydrates, Dietary Fiber
        required_cols = ['food', 'Caloric Value', 'Fat', 'Protein', 'Carbohydrates', 'Dietary Fiber']
        
        # Check if columns exist
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Missing columns {missing} in Group {i}. Skipping.")
            continue
            
        # Sample or take all? Let's take all but keep fact strings concise.
        for _, row in df.iterrows():
            fact = f"{row['food'].capitalize()} contains {row['Caloric Value']} calories, {row['Protein']}g protein, {row['Fat']}g fat, and {row['Carbohydrates']}g carbs per serving."
            if row['Dietary Fiber'] > 0:
                fact += f" It also provides {row['Dietary Fiber']}g of fiber."
            food_facts.append(fact)
            
    print(f"Extracted {len(food_facts)} food facts.")
    
    # Load existing knowledge base
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        with open(KNOWLEDGE_BASE_PATH, 'r') as f:
            kb = json.load(f)
    else:
        kb = {}
        
    # Update with new food facts
    kb['food_nutrition'] = food_facts
    
    # Save back
    with open(KNOWLEDGE_BASE_PATH, 'w') as f:
        json.dump(kb, f, indent=2)
        
    print(f"Saved updated knowledge base to {KNOWLEDGE_BASE_PATH}")

if __name__ == "__main__":
    generate_knowledge_base()
