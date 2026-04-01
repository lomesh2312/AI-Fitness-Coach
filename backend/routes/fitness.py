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
        raise HTTPException(status_code=400, detail="Height must be greater than 0")
    if req.weight <= 0:
        raise HTTPException(status_code=400, detail="Weight must be greater than 0")
    if req.age <= 0:
        raise HTTPException(status_code=400, detail="Age must be greater than 0")

    try:
        validate_fitness_input(req.age, req.weight, req.height)
        
        bmi = req.weight / (req.height ** 2)
        
        gender_num = 1 if req.gender.lower() == "male" else 0
        activity_num = 1
        
        # ML Confidence Check
        level, confidence = prediction_service.predict_fitness(bmi, req.age, gender_num, activity_num)
        
        # Core Rules Engine for Fitness Labels
        if bmi < 18.5:
            fitness_level = "Underweight"
            fitness_risk = "Moderate"
        elif bmi < 25:
            fitness_level = "Healthy"
            fitness_risk = "Low"
        elif bmi < 30:
            fitness_level = "Overweight"
            fitness_risk = "Moderate"
        else:
            fitness_level = "Obese"
            fitness_risk = "High"
            
        # Get AI or Rule-Based Coach Advice
        advice = coach_service.get_coach_advice(bmi, fitness_level, req.food_preference, req.goal)
        
        return {
            "bmi": round(bmi, 2),
            "fitness": {
                "level": fitness_level,
                "risk": fitness_risk
            },
            # THIS IS STEP 4: MANDATORY EXPLANATION ADDED TO OUTPUT
            "explanation": advice.get("explanation", "Stay consistent with your routine!"),
            "plan": {
                "summary": advice.get("summary", ""),
                "diet": advice.get("diet", ""),
                "exercise": advice.get("exercise", []),
                "yoga": advice.get("yoga", [])
            }
        }
    except ValueError as val_e:
        raise HTTPException(status_code=400, detail=str(val_e))
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
