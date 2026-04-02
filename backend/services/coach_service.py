import os
import re
import random
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ── AUTO-LOAD .env FILE (works locally; Render uses env dashboard) ──────────
# This means: if you create a .env file next to this project with
# OPENAI_API_KEY=sk-xxx  → it gets loaded here automatically.
try:
    from dotenv import load_dotenv
    load_dotenv()  # Reads .env from the project root
except ImportError:
    pass  # dotenv not installed → silently continue (Render sets env directly)

# ── LLM INITIALIZATION ──────────────────────────────────────────────────────
_api_key = os.getenv("OPENAI_API_KEY")
llm = None
if _api_key:
    print("✅ [LLM] OpenAI API key found — LLM is ACTIVE (gpt-3.5-turbo)")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
else:
    print(
        "⚠️  [LLM] WARNING: No OPENAI_API_KEY found!\n"
        "    → Locally: copy .env.example to .env and paste your key.\n"
        "    → Render:  add it in Dashboard → Environment → OPENAI_API_KEY.\n"
        "    → System will use rule-based fallback until key is provided."
    )

# ── PROMPT TEMPLATE ─────────────────────────────────────────────────────────
# The RAG section is now explicitly labelled as EVIDENCE YOU MUST USE.
# This forces the LLM to ground its advice in the retrieved facts,
# instead of generating generic responses from its training alone.
coach_prompt_template = """
You are a Senior AI Fitness Coach. Provide personalized, science-backed advice.

=== USER PROFILE ===
- BMI       : {bmi}  →  Category: {fitness_level}
- User Goal : {goal}
- Diet Type : {food_pref}

=== SCIENTIFIC EVIDENCE (YOU MUST BASE YOUR ADVICE ON THESE FACTS) ===
{rag_context}

=== YOUR TASK ===
Using the evidence above AND the user's profile, produce a highly personalized
plan. Do NOT be generic. Reference the user's BMI, goal, and food type explicitly.

1. DIET    : 3 specific meals tailored to their food preference and goal.
   Example: "Grilled Chicken & Quinoa Bowl with steamed broccoli"
2. YOGA    : 2 yoga poses with exact duration and why it helps their specific BMI.
3. EXERCISE: 2 specific workouts (sets/reps/duration) matched to their goal.
4. SUMMARY : 2 motivational sentences that mention their BMI and goal by name.

STRICT RESPONSE FORMAT — do not add extra sections:
DIET: [meal 1, meal 2, meal 3]
YOGA: [pose with duration and benefit, pose 2]
EXERCISE: [workout with sets/reps, workout 2]
SUMMARY: [2-sentence personalized motivational text]
"""

prompt = PromptTemplate(
    input_variables=[
        "bmi", "fitness_level", "food_pref", "goal", "rag_context"
    ],
    template=coach_prompt_template,
)


def _generate_human_explanation(bmi: float, level: str, goal: str) -> str:
    """Always guarantees a human explanation, regardless of LLM."""
    bmi_round = round(bmi, 2)
    parts = [
        f"Your BMI is {bmi_round}, "
        f"which places you in the **{level}** category."
    ]

    if bmi < 18.5:
        parts.append(
            "Being underweight reduces muscle mass. A caloric surplus "
            "with high-protein foods is critical."
        )
    elif bmi < 25:
        parts.append(
            "Your weight is in a healthy range! Maintaining this balance "
            "through consistent activity and nutrition is the key."
        )
    elif bmi < 30:
        parts.append(
            "Being overweight slightly elevates your risk for cardio issues. "
            "A moderate caloric deficit will yield excellent results."
        )
    else:
        parts.append(
            "Obesity significantly raises the risk of diabetes/heart disease. "
            "Start with low-impact cardio."
        )

    parts.append(
        f"Since your goal is **{goal}**, I have designed a personalized "
        "action plan below to help you achieve it."
    )
    return " ".join(parts)


class CoachService:
    def get_coach_advice(self, bmi, fitness_level, food_pref, goal):
        # 1. GENERATE MANDATORY EXPLANATION FIRST
        explanation = _generate_human_explanation(bmi, fitness_level, goal)

        # 2. TRIGGER MEMORY-SAFE RAG (Lazy loads inside retrieve)
        from rag.rag_system import rag_instance

        query = (
            f"User goal {goal}. BMI {bmi} ({fitness_level}). "
            f"Food: {food_pref}."
        )

        try:
            rag_hints = rag_instance.retrieve(query, top_k=3)
            rag_context = (
                "\n".join([f"- {fact}" for fact in rag_hints])
                if rag_hints
                else "Hydration is key."
            )
            print(f"📚 [RAG] Successfully retrieved {len(rag_hints)} facts.")
        except Exception as e:
            print(
                f"⚠️ [RAG] System failed gracefully ({e}). "
                "Falling back without context."
            )
            rag_context = (
                "Drink water and prioritize high-quality protein."
            )

        # 3. LLM AS PRIMARY INTELLIGENCE ──────────────────────────────────
        if llm:
            try:
                print(
                    f"🤖 [LLM] Calling OpenAI for: goal={goal}, "
                    f"bmi={round(bmi,2)}, food={food_pref}"
                )
                custom_prompt = prompt.format(
                    bmi=round(bmi, 2),
                    fitness_level=fitness_level,
                    food_pref=food_pref,
                    goal=goal,
                    rag_context=rag_context,
                )

                response = llm.invoke(custom_prompt)
                response_text = (
                    response.content if hasattr(response, "content")
                    else str(response)
                )
                print(f"✅ [LLM] Response received ({len(response_text)} chars)")

                parsed = self._parse_response(response_text)

                return {
                    "explanation": explanation,
                    "diet": parsed.get("diet") or "Balanced Protein Meal",
                    "yoga": parsed.get("yoga") or ["Tadasana (Mountain Pose)"],
                    "exercise": parsed.get("exercise") or ["30 min Brisk Walk"],
                    "summary": (
                        f"🎯 Goal: {goal}. "
                        + (parsed.get("summary") or "Stay consistent!")
                    ),
                    "source": "llm",  # lets you see in API response which path ran
                }
            except Exception as e:
                print(
                    f"❌ [LLM] OpenAI call failed: {e}\n"
                    "   → Check your API key and quota at platform.openai.com"
                )

        # 4. IF NO API KEY OR LLM CRASHES → SMART RANDOMIZED RULE-BASED FALLBACK
        print("🔄 [FALLBACK] No LLM — using randomized rule-based fallback plan.")
        return self._rule_based_fallback(
            fitness_level, food_pref, bmi, goal, explanation
        )

    def _parse_response(self, text):
        sections = {"diet": "", "yoga": [], "exercise": [], "summary": ""}
        for key in ["DIET", "YOGA", "EXERCISE", "SUMMARY"]:
            match = re.search(
                f"{key}:\\s*(.*?)(?=\\n[A-Z]+:|$)", text,
                re.DOTALL | re.IGNORECASE
            )
            if match:
                content = match.group(1).strip()
                if key in ["YOGA", "EXERCISE", "DIET"]:
                    # Split by commas or newlines for arrays
                    clean = [
                        item.strip()
                        for item in re.split(
                            r",|\n-", content.replace("[", "").replace("]", "")
                        )
                        if item.strip()
                    ]
                    sections[key.lower()] = clean if clean else [content]
                else:
                    sections[key.lower()] = content
        return sections

    # --- STEP 6: PREVENT REPETITION (SMART FALLBACK) ---
    def _rule_based_fallback(self, level, pref, bmi, goal, explanation):
        is_veg = "veg" in pref.lower()
        is_loss = "loss" in goal.lower() or bmi >= 25.0
        is_gain = "gain" in goal.lower() or bmi < 18.5

        # Highly varied arrays
        diets = {
            "veg_loss": [
                "Lentil Soup & Cucumber Salad",
                "Quinoa Veggie Bowl",
                "Green Tea & Sprout Salad",
                "Brown Rice & Moong Dal",
                "Oats with chia seeds",
            ],
            "nonveg_loss": [
                "Grilled Chicken & Broccoli",
                "Baked Salmon",
                "Chicken Salad with Olive Oil",
                "Boiled Eggs & Spinach",
                "Turkey Lettuce Wraps",
            ],
            "veg_gain": [
                "Tofu & Sweet Potato",
                "Soy Chunks Pulav",
                "Paneer Tikka Meal",
                "Chickpea Curry",
                "Peanut Butter Banana Shake",
            ],
            "nonveg_gain": [
                "Chicken Breast & Quinoa",
                "Steak & Roasted Potatoes",
                "Eggs & Oatmeal",
                "Turkey Pasta",
                "Greek Yogurt & Mince Meat",
            ],
            "veg_maint": [
                "Dal & Brown Rice",
                "Khichdi",
                "Mixed Veg Curry",
                "Oats & Fruit Bowl",
                "Veggie Wrap",
            ],
            "nonveg_maint": [
                "Chicken Curry & Rice",
                "Fish Tikka",
                "Egg Curry",
                "Grilled Chicken Wrap",
                "Tuna Salad",
            ],
        }

        workouts = {
            "loss": [
                "30 min Brisk Walk",
                "15 min HIIT",
                "Stair Climbing (10 min)",
                "Jump Rope (5 min)",
                "Cycling at moderate pace",
            ],
            "gain": [
                "Progressive Strength Training",
                "Heavy Weightlifting (Squats/Deadlifts)",
                "Resistance Band Exercises",
                "Pullups & Pushups",
            ],
            "maint": [
                "20 min Morning Jog",
                "Bodyweight Squats (20 reps)",
                "Core Planks (3 sets)",
                "Swimming (20 min)",
                "Light Callisthenics",
            ],
        }

        yogas = [
            "Surya Namaskar (10 rounds)",
            "Balasana (Child Pose)",
            "Trikonasana (Triangle Pose)",
            "Bhujangasana (Cobra)",
            "Vrikshasana (Tree Pose)",
        ]

        # Select the right pool
        diet_key = (
            f"{'veg' if is_veg else 'nonveg'}_"
            f"{'loss' if is_loss else 'gain' if is_gain else 'maint'}"
        )
        workout_key = "loss" if is_loss else "gain" if is_gain else "maint"

        # Randomize items so user never sees exact same plan twice
        diet_list = random.sample(
            diets[diet_key], min(3, len(diets[diet_key]))
        )
        diet_str = "\n".join([f"• {x}" for x in diet_list])

        workout_list = random.sample(workouts[workout_key], 2)
        yoga_list = random.sample(yogas, 2)

        msg = (
            f"Coach says: Based on your BMI of {round(bmi, 1)} "
            f"and {goal} goal, I've built a targeted plan. Stick to it! 🚀"
        )

        return {
            "explanation": explanation,
            "diet": diet_str,
            "yoga": yoga_list,
            "exercise": workout_list,
            "summary": msg,
        }


coach_service = CoachService()
