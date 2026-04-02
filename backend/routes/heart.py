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
                    "explanation": "🚨 EXTREME LEVELS DETECTED: Your vitals "
                    "are far outside realistic limits. Seek medical care.",
                    "warning": "⚠️ EMERGENCY: Extreme BP/Cholesterol.",
                }
            }

        validate_heart_input(req.blood_pressure, req.cholesterol)

        gender_num = 1 if req.gender.lower() == "male" else 0
        
        # ── Get blended prediction (60% Rule / 40% ML) ──────────────────
        prob_final, risk_final = prediction_service.predict_heart_risk(
            req.age, gender_num, req.bmi, req.blood_pressure, req.cholesterol
        )

        risk_pct = round(prob_final * 100, 1)  # type: ignore[call-overload]

        # ── Setup UI Labels & Tone ──────────────────────────────────────
        warning_msg = None
        if risk_final == "High":
            risk_desc = "High"
            coach_tone = (
                "Your vitals are in a high-risk zone. "
                "Strategic lifestyle changes are strongly recommended."
            )
            warning_msg = "⚠️ Medical consultation recommended"
        elif risk_final == "Medium":
            risk_desc = "Moderate"
            coach_tone = (
                "You are leaning towards a risk zone. "
                "Prevention now will save you later."
            )
        else:
            risk_desc = "Low"
            coach_tone = "Excellent! Your vitals are within healthy ranges."

        # ── Construct Scientific Explanation ────────────────────────────
        bp_sev = "high" if req.blood_pressure >= 140 else "moderate" if req.blood_pressure >= 130 else "normal"
        chol_sev = "high" if req.cholesterol >= 240 else "moderate" if req.cholesterol >= 200 else "normal"

        if risk_desc == "Low":
            explanation = (
                f"Heart risk is Low ({risk_pct}%). BP ({req.blood_pressure}) "
                f"and Chol ({req.cholesterol}) are healthy for your profile."
            )
        else:
            reasons = []
            if chol_sev != "normal":
                reasons.append(f"Elevated cholesterol ({req.cholesterol} mg/dL)")
            if bp_sev != "normal":
                reasons.append(f"High blood pressure ({req.blood_pressure} mmHg)")
            
            explanation = (
                f"Your risk is {risk_desc} ({risk_pct}%). This is primarily "
                f"due to: {', '.join(reasons)}. "
                "We recommend focusing on cardiovascular health improvements."
            )

        # ── Personalization ─────────────────────────────────────────────
        bmi_cat = "underweight" if req.bmi < 18.5 else "overweight" if req.bmi > 25 else "healthy"
        personalization = (
            f"\n\n🩺 Coach Note: {coach_tone} At {req.age} with a {bmi_cat} BMI, "
            f"your goal to '{req.goal}' is very achievable with consistency."
        )

        return {
            "heart_risk": {
                "probability": prob_final,
                "category": risk_desc,
                "explanation": explanation + personalization,
                "warning": warning_msg,
            }
        }

    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        print(f"Error handling request: {e}")
        raise HTTPException(
            status_code=500, detail="Server error. Please try again."
        )
