from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from services.prediction_service import prediction_service

router = APIRouter()

class HeartRequest(BaseModel):
    age:            int
    gender:         str
    bmi:            float
    blood_pressure: float
    cholesterol:    float
    goal:           str = "Maintain Fitness"

    @validator("blood_pressure")
    def bp_must_be_valid(cls, v):
        if v <= 0:
            raise ValueError("Blood pressure must be greater than 0")
        if v > 300:
            raise ValueError("Blood pressure exceeds realistic maximum (300 mmHg)")
        return v

    @validator("cholesterol")
    def chol_must_be_valid(cls, v):
        if v <= 0:
            raise ValueError("Cholesterol must be greater than 0")
        if v > 700:
            raise ValueError("Cholesterol exceeds realistic maximum (700 mg/dL)")
        return v

    @validator("bmi")
    def bmi_must_be_positive(cls, v):
        if v <= 0 or v > 80:
            raise ValueError("BMI must be between 1 and 80")
        return v

    @validator("age")
    def age_must_be_valid(cls, v):
        if v <= 0 or v > 120:
            raise ValueError("Age must be between 1 and 120")
        return v


def _bp_status(bp: float) -> str:
    if bp >= 140: return "high"
    if bp >= 130: return "moderate"
    return "safe"

def _chol_status(chol: float) -> str:
    if chol >= 240: return "high"
    if chol >= 200: return "moderate"
    return "safe"

def _bmi_label(bmi: float) -> str:
    if bmi < 18.5: return "underweight"
    if bmi < 25:   return "a healthy weight"
    if bmi < 30:   return "overweight"
    return "obese"

def _build_explanation(risk_desc: str, risk_pct: float, bp: float, chol: float,
                        bpi: str, choli: str, age: int, bmi: float, goal: str,
                        coach_tone: str) -> str:
    """Build a full human-language explanation for the heart risk result."""

    # ── Opening ──────────────────────────────────────────────────────────────
    parts = [f"Your heart disease risk is **{risk_desc}** ({risk_pct}% probability)."]

    if risk_desc == "Low":
        parts.append(
            f"Your Blood Pressure ({bp} mmHg) and Cholesterol ({chol} mg/dL) are both "
            f"within safe ranges. This is excellent news — keep maintaining your healthy habits."
        )
    else:
        # ── Cholesterol reasoning ─────────────────────────────────────────────
        if choli == "high":
            parts.append(
                f"Your cholesterol is **{chol} mg/dL**, which is above the recommended limit of 200 mg/dL. "
                f"High cholesterol causes fatty deposits (plaque) to build up inside your arteries, "
                f"narrowing them and making it harder for your heart to pump blood efficiently."
            )
        elif choli == "moderate":
            parts.append(
                f"Your cholesterol ({chol} mg/dL) is borderline-high (200–239 mg/dL). "
                f"If left unchecked, this can contribute to plaque buildup in the arteries over time."
            )

        # ── BP reasoning ──────────────────────────────────────────────────────
        if bpi == "high":
            parts.append(
                f"Your Blood Pressure at **{bp} mmHg** is elevated (≥140 mmHg = Stage 2 Hypertension). "
                f"High blood pressure puts constant extra stress on your heart and artery walls, "
                f"significantly raising heart attack and stroke risk."
            )
        elif bpi == "moderate":
            parts.append(
                f"Your Blood Pressure ({bp} mmHg) is in the elevated range (130–139 mmHg). "
                f"While not yet Stage 2 hypertension, this is a signal to reduce salt, stress, "
                f"and increase aerobic exercise."
            )

        # ── BMI note ──────────────────────────────────────────────────────────
        bmi_txt = _bmi_label(bmi)
        parts.append(
            f"Combined with your BMI of {round(bmi, 1)} ({bmi_txt}) at age {age}, "
            f"these factors compound each other and raise your overall risk."
        )

        # ── Action ───────────────────────────────────────────────────────────
        if risk_desc == "Critical":
            parts.append("⚠️ **Immediate medical consultation is strongly advised.** Do not delay.")
        elif risk_desc == "High":
            parts.append(
                "It is strongly recommended to consult a doctor and begin immediate lifestyle changes: "
                "reduce sodium, increase fibre intake, and aim for 30 min of cardio 5 days per week."
            )
        elif risk_desc == "Moderate":
            parts.append(
                "Consider improving your diet (less processed food, more vegetables), "
                "reducing salt intake, quitting smoking if applicable, and increasing physical activity."
            )

    # ── Personalised coach note ───────────────────────────────────────────────
    parts.append(
        f"\n\n🩺 **Coach's Note:** {coach_tone} "
        f"Your goal of '{goal}' is a strong starting point — use it as your daily motivation."
    )

    return " ".join(parts)


@router.post("/heart")
async def heart_health_check(req: HeartRequest):
    try:
        gender_num = 1 if req.gender.lower() == "male" else 0
        prob_final, risk_final = prediction_service.predict_heart_risk(
            req.age, gender_num, req.bmi, req.blood_pressure, req.cholesterol
        )

        risk_pct = round(prob_final * 100, 1)

        # Enforce critical threshold override
        if req.blood_pressure >= 180 or req.cholesterol >= 300:
            risk_final = "Critical"

        bpi  = _bp_status(req.blood_pressure)
        choli = _chol_status(req.cholesterol)

        # ── Map risk to display strings ───────────────────────────────────────
        if risk_final == "Critical":
            risk_desc   = "Critical"
            coach_tone  = "This is serious — immediate medical intervention is needed. Please see a doctor today."
            warning_msg = "🚨 CRITICAL: Emergency medical attention required"
        elif risk_final == "High":
            risk_desc   = "High"
            coach_tone  = "Your health indicators need prompt attention. Act now before this escalates."
            warning_msg = "⚠️ High Risk: Medical attention strongly recommended"
        elif risk_final == "Medium":
            risk_desc   = "Moderate"
            coach_tone  = "You are in the warning zone. Small, consistent changes now will prevent big problems later."
            warning_msg = "⚠️ Moderate Risk: Consider lifestyle changes and monitoring"
        else:
            risk_desc   = "Low"
            coach_tone  = "Great job keeping your metrics in a healthy range!"
            warning_msg = None

        explanation = _build_explanation(
            risk_desc, risk_pct,
            req.blood_pressure, req.cholesterol,
            bpi, choli,
            req.age, req.bmi, req.goal, coach_tone
        )

        return {
            "heart_risk": {
                "probability": prob_final,
                "category":    risk_desc,
                "explanation": explanation,
                "warning":     warning_msg,
                "metrics": {
                    "bp_status":   bpi,
                    "chol_status": choli,
                    "bmi_label":   _bmi_label(req.bmi),
                },
            }
        }

    except ValueError as val_e:
        raise HTTPException(status_code=422, detail=str(val_e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [HEART] Unhandled error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again.")
