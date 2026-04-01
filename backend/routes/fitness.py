from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from services.prediction_service import prediction_service
from services.coach_service import coach_service

router = APIRouter()

class FitnessRequest(BaseModel):
    age:             int
    height:          float
    weight:          float
    gender:          str
    food_preference: str
    goal:            str = "Maintain Fitness"

    @validator("height")
    def height_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Height must be greater than 0")
        if v > 3.0:
            raise ValueError("Height seems unrealistic (> 3 m). Please check your units.")
        return v

    @validator("weight")
    def weight_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Weight must be greater than 0")
        if v > 500:
            raise ValueError("Weight seems unrealistic (> 500 kg). Please check.")
        return v

    @validator("age")
    def age_must_be_valid(cls, v):
        if v <= 0 or v > 120:
            raise ValueError("Age must be between 1 and 120")
        return v

    @validator("food_preference")
    def food_pref_must_be_valid(cls, v):
        allowed = {"veg", "non-veg", "nonveg", "non_veg", "vegan", "vegetarian"}
        if v.lower().replace("-", "").replace(" ", "") not in {a.replace("-", "") for a in allowed}:
            raise ValueError(f"food_preference must be 'veg' or 'non-veg', got: '{v}'")
        return v


# ── Fitness level from BMI (standard WHO categories) ────────────────────────
def _bmi_category(bmi: float) -> tuple:
    """Returns (fitness_level, fitness_risk, bmi_label)."""
    if bmi < 18.5:
        return "Underweight", "Moderate", "Underweight"
    if bmi < 25.0:
        return "Normal",      "Low",      "Normal weight"
    if bmi < 30.0:
        return "Overweight",  "Moderate", "Overweight"
    return     "Obese",       "High",     "Obese"


@router.post("/fitness")
async def fitness_analysis(req: FitnessRequest):
    try:
        bmi = req.weight / (req.height ** 2)

        fitness_level, fitness_risk, bmi_label = _bmi_category(bmi)

        gender_num   = 1 if req.gender.lower() == "male" else 0
        activity_num = 1

        # Run the ML prediction (used for confidence score only; BMI drives category)
        try:
            ml_level, confidence = prediction_service.predict_fitness(
                bmi, req.age, gender_num, activity_num
            )
        except Exception as e:
            print(f"⚠️  [ML] Prediction failed: {e}")
            ml_level, confidence = fitness_level, 0.0

        advice = coach_service.get_coach_advice(
            bmi, fitness_level, req.food_preference, req.goal
        )

        return {
            "bmi":          round(bmi, 2),
            "bmi_label":    bmi_label,
            "fitness": {
                "level":      fitness_level,
                "risk":       fitness_risk,
                "confidence": round(float(confidence), 3),
            },
            "explanation":  advice.get("explanation", ""),
            "plan": {
                "summary":  advice["summary"],
                "diet":     advice["diet"],
                "exercise": advice["exercise"],
                "yoga":     advice["yoga"],
            },
        }

    except ValueError as val_e:
        raise HTTPException(status_code=422, detail=str(val_e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [FITNESS] Unhandled error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again.")
