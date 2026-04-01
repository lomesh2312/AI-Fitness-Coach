from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.validation import validate_heart_input
from services.prediction_service import prediction_service

router = APIRouter()

class HeartRequest(BaseModel):
    age: int
    gender: str
    bmi: float
    blood_pressure: float 
    cholesterol: float
    goal: str = "Maintain Fitness"

@router.post("/heart")
async def heart_health_check(req: HeartRequest):
    try:
        if req.blood_pressure > 300 or req.cholesterol > 600:
            return {
                "heart_risk": {
                    "probability": 1.0,
                    "category": "High",
                    "explanation": "🚨 EXTREME LEVELS DETECTED: Your vitals are outside realistic limits. Please seek immediate medical attention.",
                    "warning": "⚠️ EMERGENCY: Extreme BP/Cholesterol readings."
                }
            }

        validate_heart_input(req.blood_pressure, req.cholesterol)
        
        gender_num = 1 if req.gender.lower() == "male" else 0
        prob_final, risk_final = prediction_service.predict_heart_risk(
            req.age, gender_num, req.bmi, req.blood_pressure, req.cholesterol
        )
        
        if req.blood_pressure >= 180 or req.cholesterol >= 300:
            risk_final = "Critical"
            
        risk_pct = round(prob_final * 100, 1)
        
        bp_status = "safe"
        if req.blood_pressure >= 140: bp_status = "high"
        elif req.blood_pressure >= 130: bp_status = "moderate"
        
        chol_status = "safe"
        if req.cholesterol >= 240: chol_status = "high"
        elif req.cholesterol >= 200: chol_status = "moderate"
        
        if risk_final == "Critical":
            risk_desc = "Critical"
            coach_tone = "This is serious. Immediate medical intervention is required."
            warning_msg = "⚠️ Immediate doctor consultation required"
        elif risk_final == "High":
            risk_desc = "High"
            coach_tone = "Your health indicators need immediate attention. It’s important to take action now."
            warning_msg = "⚠️ Medical attention recommended"
        elif risk_final == "Medium":
            risk_desc = "Moderate"
            coach_tone = "You’re close to a risk zone. Small changes can make a big difference."
            warning_msg = None
        else:
            risk_desc = "Low"
            coach_tone = "Great job maintaining your health!"
            warning_msg = None
            
        if risk_desc == "Low":
            explanation = f"Your heart disease risk is Low ({risk_pct}% probability). Your Blood Pressure ({req.blood_pressure} mmHg) and Cholesterol ({req.cholesterol} mg/dL) are within safe limits. To maintain this, continue your balanced diet and regular physical activity."
        else:
            bp_reason = ""
            chol_reason = ""
            
            if chol_status in ["high", "moderate"]:
                chol_reason = f"Your cholesterol level is {req.cholesterol} mg/dL, which is higher than the recommended limit. High cholesterol can lead to fat buildup in your arteries, increasing the risk of heart disease."
            if bp_status in ["high", "moderate"]:
                if chol_status in ["high", "moderate"]:
                    bp_reason = f"Additionally, your blood pressure is {req.blood_pressure} mmHg, which puts extra strain on your heart."
                else:
                    bp_reason = f"Your blood pressure is {req.blood_pressure} mmHg, which is higher than the recommended limit. High blood pressure puts extra strain on your heart and arteries, increasing cardiovascular risk."
                    
            vitals_reasoning = f"{chol_reason} {bp_reason}".strip()
            
            action = ""
            if risk_desc == "Critical":
                 action = "This puts you at serious risk of heart disease. Immediate medical consultation is strongly advised."
            elif risk_desc == "High":
                 action = "It is strongly recommended to consult a doctor and make immediate lifestyle changes."
            elif risk_desc == "Moderate":
                 action = "Consider improving your diet, reducing salt intake, and increasing physical activity."
                 
            explanation = f"Your heart disease risk is {risk_desc} ({risk_pct}% probability). {vitals_reasoning} {action}"
        
        if req.bmi < 18.5:
             bmi_str = "an underweight BMI"
        elif req.bmi > 25:
             bmi_str = "an overweight BMI"
        else:
             bmi_str = "a healthy BMI"
             
        personalization = f"\n\n🩺 Coach's Note: {coach_tone} For your age ({req.age}) with {bmi_str}, focusing on your goal to '{req.goal}' will be key to managing your long-term heart health."
        
        explanation += personalization

        return {
            "heart_risk": {
                "probability": prob_final,
                "category": risk_desc,
                "explanation": explanation,
                "warning": warning_msg
            }
        }

    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        print(f"Error handling request: {e}")
        raise HTTPException(status_code=500, detail="Server temporarily unavailable. Please try again.")
