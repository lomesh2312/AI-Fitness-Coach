import os
import re
import gc
import random
import threading
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ── LLM Initialization ──────────────────────────────────────────────────────
llm = None
_llm_lock = threading.Lock()

def _get_llm():
    global llm
    if llm is not None:
        return llm
    with _llm_lock:
        if llm is not None:
            return llm
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            print("⚠️  [LLM] OPENAI_API_KEY not set — using rule-based fallback.")
            return None
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=key,
                max_tokens=512,
            )
            print("✅ [LLM] ChatOpenAI initialized successfully.")
            return llm
        except Exception as e:
            print(f"❌ [LLM] Initialization failed: {e}")
            return None

# ── Prompt Template ──────────────────────────────────────────────────────────
_COACH_PROMPT = """You are a Senior AI Fitness & Nutrition Coach. Be motivational, specific, and human.

USER PROFILE:
- BMI: {bmi} ({bmi_label})
- Fitness Level: {fitness_level}
- Goal: {goal}
- Food Preference: {food_pref}
- Strategy: {strategy}

SCIENTIFIC CONTEXT:
{rag_context}

RULES:
- Diet: exactly 3 specific meals (no generic items like "chicken", say "Grilled Lemon Chicken with Quinoa")
- Yoga: 2 poses with duration and benefit
- Exercise: 2 exercises with sets/duration
- Summary: 2-sentence motivational coach note using the user's actual goal and BMI

FORMAT (strict):
DIET: [meal1, meal2, meal3]
YOGA: [pose1 (duration - benefit), pose2 (duration - benefit)]
EXERCISE: [exercise1 (sets/duration), exercise2 (sets/duration)]
SUMMARY: [text]
"""

_prompt = PromptTemplate(
    input_variables=["bmi", "bmi_label", "fitness_level", "food_pref", "goal", "strategy", "rag_context"],
    template=_COACH_PROMPT
)

# ── Diverse Diet Banks ───────────────────────────────────────────────────────
_DIETS = {
    "veg": {
        "underweight": [
            "Paneer Paratha + Sweet Potato Mash", "Avocado Toast + Almond Butter",
            "Chickpea Curry + Quinoa", "Energy Shake with Oats, Banana & Peanut Butter",
            "Tofu Stir Fry + Brown Rice", "Rajma + Roti + Greek Yogurt"
        ],
        "loss": [
            "Moong Dal Soup + Cucumber Salad", "Quinoa Tabbouleh + Roasted Chickpeas",
            "Green Smoothie + Flaxseed Crackers", "Steamed Broccoli + Brown Rice + Lentils",
            "Sprout Salad + Lemon Dressing", "Idli + Sambar + Coconut Chutney"
        ],
        "gain": [
            "Soy Chunks + Brown Rice + Ghee", "Paneer Tikka + Sweet Potato",
            "Dal Makhani + Roti + Curd", "Chickpea Stew + Whole Wheat Bread",
            "Banana Protein Shake + Mixed Nuts", "Rajma Masala + Jeera Rice"
        ],
        "maintain": [
            "Dal + Brown Rice + Papad", "Khichdi with Mixed Vegetables",
            "Mixed Veg Sabzi + 2 Rotis", "Oats Upma + Boiled Egg (optional)",
            "Palak Paneer + Jowar Roti", "Vegetable Poha + Coconut Water"
        ],
        "obese": [
            "Methi Thepla + Low-fat Curd", "Cucumber Raita + Multigrain Roti",
            "Stir Fried Vegetables + Millets", "Oat Bran Porridge + Berries",
            "Beetroot Salad + Lemon Juice", "Vegetable Daliya + Buttermilk"
        ],
    },
    "nonveg": {
        "underweight": [
            "Eggs & Bacon + Whole Wheat Toast", "Chicken Stew + Brown Rice",
            "Tuna Sandwich on Multigrain Bread", "Beef Steak + Mashed Sweet Potato",
            "Protein Shake + Boiled Eggs + Almonds", "Salmon + Quinoa + Steamed Broccoli"
        ],
        "loss": [
            "Grilled Chicken Breast + Asparagus", "Baked Salmon + Steamed Spinach",
            "Boiled Eggs + Mixed Green Salad", "Chicken Soup + Multigrain Crackers",
            "Tuna Salad + Olive Oil Dressing", "Turkey Lettuce Wraps + Salsa"
        ],
        "gain": [
            "Chicken Breast + Quinoa + Ghee", "Eggs & Oatmeal + Peanut Butter",
            "Steak + Roasted Potatoes", "Turkey Slices + Brown Rice",
            "Salmon + Avocado + Sweet Potato", "Mutton Curry + Roti + Curd"
        ],
        "maintain": [
            "Chicken Curry + Rice", "Fish Tikka + Stir-Fried Veggies",
            "Egg Curry + 2 Rotis", "Grilled Chicken Wrap + Greek Yogurt",
            "Keema Paratha + Salad", "Prawn Stir Fry + Brown Rice"
        ],
        "obese": [
            "Grilled Fish + Steamed Veggies", "Chicken Salad + Olive Oil",
            "Boiled Eggs + Cucumber Slices", "Baked Chicken + Zucchini",
            "Egg White Omelette + Spinach", "Steamed Fish + Brown Rice (small portion)"
        ],
    }
}

_YOGA_POOL = [
    "Tadasana (Mountain Pose) — 2 min — improves posture and body awareness",
    "Bhujangasana (Cobra Pose) — 30s × 3 — improves spinal flexibility",
    "Virabhadrasana I (Warrior I) — 1 min each side — builds lower body strength",
    "Balasana (Child Pose) — 3 min — reduces stress and lower back tension",
    "Surya Namaskar (Sun Salutation) — 10 rounds — full-body activation",
    "Trikonasana (Triangle Pose) — 1 min each side — stretches hips and spine",
    "Setu Bandhasana (Bridge Pose) — 30s × 3 — strengthens glutes and back",
    "Anulom Vilom (Alternate Breathing) — 5 min — reduces BP and anxiety",
    "Paschimottanasana (Seated Forward Bend) — 1 min — stretches hamstrings",
    "Vrikshasana (Tree Pose) — 1 min each side — improves balance and focus",
]

_WORKOUTS = {
    "underweight": [
        "Progressive Overload Weight Training — 4 sets of 8 reps",
        "Dumbbell Bicep Curls + Shoulder Press — 3 sets of 10 reps",
        "Resistance Band Pull-Aparts — 3 sets of 15 reps",
        "Bodyweight Pullups + Pushups Circuit — 3 sets",
    ],
    "loss": [
        "30 min Brisk Walking at 6 km/h daily",
        "HIIT: 20 sec sprint + 40 sec rest × 10 rounds",
        "Stair Climbing for 15 minutes",
        "Cycling at moderate pace — 45 min",
        "Jump Rope — 500 skips per session",
    ],
    "gain": [
        "Heavy Compound Lifts (Squat/Deadlift/Bench) — 4 sets of 6 reps",
        "Resistance Band Full-Body Circuit — 3 sets",
        "Dumbbell Rows + Lunges — 3 sets of 8 reps",
        "Progressive Push-up Challenge — 5 sets max reps",
    ],
    "maintain": [
        "20 min Morning Jog at comfortable pace",
        "Core Plank Hold — 3 × 60 seconds",
        "Bodyweight Squats — 3 × 20 reps",
        "Swimming for 30 minutes — moderate pace",
        "Yoga Flow + Light Stretching — 30 min",
    ],
    "obese": [
        "Low-impact Walking — 20 min, twice daily",
        "Chair Squats — 3 sets of 10 reps",
        "Seated Resistance Band Exercises — 3 sets",
        "Water Aerobics or Swimming — 30 min",
    ],
}

def _bmi_label(bmi):
    if bmi < 18.5: return "Underweight"
    if bmi < 25:   return "Normal weight"
    if bmi < 30:   return "Overweight"
    return "Obese"

def _diet_key(bmi, goal):
    goal_lower = goal.lower()
    if bmi < 18.5:          return "underweight"
    if bmi >= 30:           return "obese"
    if "loss" in goal_lower: return "loss"
    if "gain" in goal_lower: return "gain"
    return "maintain"

def _workout_key(bmi, goal):
    goal_lower = goal.lower()
    if bmi < 18.5:           return "underweight"
    if bmi >= 30:            return "obese"
    if "loss" in goal_lower: return "loss"
    if "gain" in goal_lower: return "gain"
    return "maintain"

def _make_explanation(bmi, fitness_level, goal):
    label = _bmi_label(bmi)
    parts = [f"Your BMI is {round(bmi, 2)}, placing you in the **{label}** category."]
    if bmi < 18.5:
        parts.append("Being underweight means your body may lack key nutrients and muscle mass. Increasing caloric intake through nutritious, protein-rich foods is essential for healthy weight gain.")
    elif bmi < 25:
        parts.append("Your weight is in the healthy range. The focus should be on maintaining this balance through consistent exercise and a varied, nutritious diet.")
    elif bmi < 30:
        parts.append("Being overweight increases the risk of cardiovascular disease and type 2 diabetes. A moderate calorie deficit combined with daily physical activity will help bring your BMI into the healthy range.")
    else:
        parts.append("Obesity significantly increases health risks. Even a 5–10% reduction in body weight can dramatically improve blood pressure, blood sugar, and cholesterol levels.")
    parts.append(f"Given your goal of **{goal}**, your personalised plan below is designed to target this specifically.")
    return " ".join(parts)

class CoachService:
    def get_coach_advice(self, bmi, fitness_level, food_pref, goal):
        # Lazy-import rag to avoid circular import / startup crash
        from rag.rag_system import rag_instance

        dk = _diet_key(bmi, goal)
        wk = _workout_key(bmi, goal)
        pref = "veg" if food_pref.lower() == "veg" else "nonveg"
        label = _bmi_label(bmi)

        query = f"goal={goal}, BMI={round(bmi,1)} ({label}), food={food_pref}, fitness={fitness_level}"
        try:
            rag_hints = rag_instance.retrieve(query, top_k=4)
            print(f"📚 [RAG] Retrieved {len(rag_hints)} facts.")
        except Exception as e:
            print(f"⚠️  [RAG] retrieval failed: {e}")
            rag_hints = []
        rag_context = "\n".join([f"- {f}" for f in rag_hints]) if rag_hints else "No additional context available."

        # ── Try LLM first ───────────────────────────────────────────────────
        active_llm = _get_llm()
        if active_llm:
            try:
                strategy_map = {
                    "underweight": "caloric surplus + strength training",
                    "loss":        "caloric deficit + cardio",
                    "gain":        "caloric surplus + resistance training",
                    "maintain":    "balanced nutrition + moderate activity",
                    "obese":       "low-impact activity + portion control",
                }
                formatted = _prompt.format(
                    bmi=round(bmi, 2),
                    bmi_label=label,
                    fitness_level=fitness_level,
                    food_pref=food_pref,
                    goal=goal,
                    strategy=strategy_map.get(dk, "balanced approach"),
                    rag_context=rag_context,
                )
                # Use .invoke() — compatible with LangChain 0.1 and 0.2+
                response = active_llm.invoke(formatted)
                response_text = response.content if hasattr(response, "content") else str(response)
                print(f"✅ [LLM] Response received ({len(response_text)} chars).")
                parsed = self._parse_response(response_text)
                parsed["explanation"] = _make_explanation(bmi, fitness_level, goal)
                gc.collect()
                return parsed
            except Exception as e:
                print(f"❌ [LLM] Error during invoke: {e}")

        # ── Rule-based fallback ──────────────────────────────────────────────
        print("🔄 [FALLBACK] Using rule-based response.")
        return self._rule_based_fallback(bmi, pref, dk, wk, goal, fitness_level)

    def _parse_response(self, text):
        sections = {"diet": "", "yoga": [], "exercise": [], "summary": ""}
        patterns = {
            "DIET":     ("diet",     False),
            "YOGA":     ("yoga",     True),
            "EXERCISE": ("exercise", True),
            "SUMMARY":  ("summary",  False),
        }
        for key, (field, is_list) in patterns.items():
            match = re.search(rf"{key}:\s*(.*?)(?=\n(?:DIET|YOGA|EXERCISE|SUMMARY):|$)", text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip().strip("[]")
                if is_list:
                    items = [i.strip() for i in re.split(r",(?![^(]*\))", content) if i.strip()]
                    sections[field] = items if items else [content]
                else:
                    sections[field] = content
        return sections

    def _rule_based_fallback(self, bmi, pref, dk, wk, goal, fitness_level):
        diet_options = _DIETS[pref].get(dk, _DIETS[pref]["maintain"])
        diet_items   = random.sample(diet_options, min(3, len(diet_options)))
        diet         = "\n".join([f"• {item}" for item in diet_items])

        yoga_items   = random.sample(_YOGA_POOL, 2)
        workout_pool = _WORKOUTS.get(wk, _WORKOUTS["maintain"])
        workout      = random.sample(workout_pool, min(2, len(workout_pool)))

        label = _bmi_label(bmi)
        tone_map = {
            "underweight": f"Your BMI of {round(bmi,1)} ({label}) means gaining healthy weight is the priority. Focus on calorie-dense, nutritious foods and progressive strength training.",
            "loss":        f"At BMI {round(bmi,1)} ({label}), consistent cardio and a moderate calorie deficit will get you results. Small daily wins build big long-term changes.",
            "gain":        f"To build muscle at BMI {round(bmi,1)}, eat in a caloric surplus and push heavier weights each week. Recovery and sleep are just as important as training.",
            "maintain":    f"You're at a healthy BMI of {round(bmi,1)}. The goal now is consistency — keep moving, keep eating well, and focus on long-term habits.",
            "obese":       f"Starting from a BMI of {round(bmi,1)}, every step matters. Low-impact activity and portion control will create safe, sustainable fat loss.",
        }
        coach_note = tone_map.get(dk, f"BMI {round(bmi,1)}: stay consistent with your nutrition and movement plan.")

        return {
            "diet":        diet,
            "yoga":        yoga_items,
            "exercise":    workout,
            "summary":     f"🏋️ Coach says: {coach_note} Keep following this plan and track your progress weekly! 🚀",
            "explanation": _make_explanation(bmi, fitness_level, goal),
        }

coach_service = CoachService()
