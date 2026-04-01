import torch
import joblib
import os
import numpy as np
from models.fitness_model import FitnessModel
from models.disease_model import DiseaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")


class PredictionService:
    def __init__(self):
        print("--- Initializing Prediction Service ---")
        self.fit_model = FitnessModel()
        fit_path = os.path.join(CHECKPOINT_DIR, "fitness_model.pth")
        if os.path.exists(fit_path):
            print(f"Loading Fitness Model from {fit_path}")
            self.fit_model.load_state_dict(torch.load(fit_path, weights_only=True))
            self.fit_model.eval()

        self.dis_model = DiseaseModel(input_size=5)
        dis_path = os.path.join(CHECKPOINT_DIR, "disease_model.pth")
        if os.path.exists(dis_path):
            print(f"Loading Disease Model from {dis_path}")
            self.dis_model.load_state_dict(torch.load(dis_path, weights_only=True))
            self.dis_model.eval()

        self.fit_scaler = (
            joblib.load(os.path.join(CHECKPOINT_DIR, "fitness_scaler.pkl"))
            if os.path.exists(os.path.join(CHECKPOINT_DIR, "fitness_scaler.pkl"))
            else None
        )
        self.dis_scaler = (
            joblib.load(os.path.join(CHECKPOINT_DIR, "disease_scaler.pkl"))
            if os.path.exists(os.path.join(CHECKPOINT_DIR, "disease_scaler.pkl"))
            else None
        )
        print("Prediction Service Ready ✅")

    def predict_fitness(self, bmi, age, gender_num, activity_num):
        features = np.array([[bmi, age, gender_num, activity_num]])
        if self.fit_scaler:
            features = self.fit_scaler.transform(features)

        tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            probs = self.fit_model.get_confidence(tensor)
            conf, idx = torch.max(probs, dim=1)

        level = {0: "Poor", 1: "Average", 2: "Fit"}.get(idx.item(), "Average")
        return level, float(conf.item())

    def predict_heart_risk(self, age, gender_num, bmi, bp, cholesterol):
        features = np.array([[age, gender_num, bmi, bp, cholesterol]])
        if self.dis_scaler:
            features = self.dis_scaler.transform(features)

        tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            prob = self.dis_model(tensor).item()

        risk_ml = "Low"
        if prob > 0.7:
            risk_ml = "High"
        elif prob > 0.3:
            risk_ml = "Medium"

        risk_final, prob_final = self._apply_rules_engine(
            risk_ml, prob, bmi, bp, cholesterol
        )

        print(f"--- Heart Risk Debug ---")
        print(f"Input: BP={bp}, Chol={cholesterol}, BMI={bmi}")
        print(f"ML Prediction: {risk_ml} (Prob: {prob:.2f})")
        print(f"Final Outcome: {risk_final} (Prob: {prob_final:.2f})")
        print(f"------------------------")

        return float(prob_final), risk_final

    def _apply_rules_engine(self, risk_ml, prob_ml, bmi, bp, cholesterol):
        bp_sev = "Low"
        if bp >= 140:
            bp_sev = "High"
        elif bp >= 130:
            bp_sev = "Moderate"

        chol_sev = "Low"
        if cholesterol >= 240:
            chol_sev = "High"
        elif cholesterol >= 200:
            chol_sev = "Moderate"

        risk_final = risk_ml
        prob_final = prob_ml

        if bp >= 180 or cholesterol >= 300:
            risk_final = "High"
            prob_final = max(0.95, prob_ml)
        elif bp >= 130 and cholesterol >= 200:
            risk_final = "High"
            prob_final = max(0.80, prob_ml)
        elif bp_sev == "High" or chol_sev == "High":
            risk_final = "High"
            prob_final = max(0.75, prob_ml)
        elif bp_sev == "Moderate" or chol_sev == "Moderate":
            risk_final = "Medium"
            prob_final = max(0.55, prob_ml)
        elif bp_sev == "Low" and chol_sev == "Low":
            risk_final = "Low"
            prob_final = min(0.30, prob_ml)

        return risk_final, prob_final


prediction_service = PredictionService()
