from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.validation import validate_fitness_input
from services.prediction_service import prediction_service
from services.coach_service import coach_service

router = APIRouter()


class FitnessRequest(BaseModel):
    age: int
    height: float
    weight: float
    gender: str
    food_preference: str
    goal: str = "Maintain Fitness"


@router.post("/fitness")
async def fitness_analysis(req: FitnessRequest):
    if req.height <= 0:
        raise HTTPException(
            status_code=400, detail="Height must be greater than 0"
        )
    if req.weight <= 0:
        raise HTTPException(
            status_code=400, detail="Weight must be greater than 0"
        )
    if req.age <= 0:
        raise HTTPException(
            status_code=400, detail="Age must be greater than 0"
        )

    try:
        validate_fitness_input(req.age, req.weight, req.height)

        bmi = req.weight / (req.height**2)

        gender_num = 1 if req.gender.lower() == "male" else 0
        activity_num = 1

        # ── STEP 1: Get ML model prediction ──────────────────────────────
        # predict_fitness() returns (level, confidence)
        # level      → e.g. "Poor", "Average", "Fit"  (from the trained model)
        # confidence → e.g. 0.87  (87% sure — how much to trust the model)
        ml_level, ml_confidence = prediction_service.predict_fitness(
            bmi, req.age, gender_num, activity_num
        )

        # ── STEP 2: BMI rule-based label (fallback when ML is uncertain) ─────
        if bmi < 18.5:
            bmi_label = "Underweight"
            fitness_risk = "Moderate"
        elif bmi < 25:
            bmi_label = "Healthy"
            fitness_risk = "Low"
        elif bmi < 30:
            bmi_label = "Overweight"
            fitness_risk = "Moderate"
        else:
            bmi_label = "Obese"
            fitness_risk = "High"

        # ── STEP 3: DECIDE which label to use ───────────────────────────
        # Rule: if ML model is >= 60% confident, trust the model.
        # Otherwise, fall back to the simple BMI rule.
        # This means: if the model is uncertain, we don’t blindly follow it.
        ML_CONFIDENCE_THRESHOLD = 0.60
        if ml_confidence >= ML_CONFIDENCE_THRESHOLD:
            fitness_level = ml_level
            level_source = f"ML model ({ml_confidence*100:.0f}% confident)"
        else:
            fitness_level = bmi_label
            level_source = (
                f"BMI rule (ML confidence too low: {ml_confidence*100:.0f}%)"
            )
        print(
            f"🏋️  [FITNESS] BMI={bmi:.2f} | "
            f"ML='{ml_level}'({ml_confidence:.2f}) | "
            f"Used='{fitness_level}' via {level_source}"
        )

        # ── STEP 4: Get AI Coach advice ───────────────────────────────
        advice = coach_service.get_coach_advice(
            bmi, fitness_level, req.food_preference, req.goal
        )

        return {
            "bmi": round(bmi, 2),  # type: ignore[call-overload]
            "fitness": {
                "level": fitness_level,
                "risk": fitness_risk,
                "ml_confidence": round(ml_confidence, 3),  # expose ML confidence
                "level_source": level_source,               # ML or BMI-rule?
            },
            "explanation": advice.get(
                "explanation", "Stay consistent with your routine!"
            ),
            "plan": {
                "summary": advice.get("summary", ""),
                "diet": advice.get("diet", ""),
                "exercise": advice.get("exercise", []),
                "yoga": advice.get("yoga", []),
                "source": advice.get("source", "fallback"),  # llm or fallback?
            },
        }
    except ValueError as val_e:
        raise HTTPException(status_code=400, detail=str(val_e))
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
