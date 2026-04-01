import os
import re
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from rag.rag_system import rag_instance

llm = None
if os.getenv("OPENAI_API_KEY"):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

coach_prompt_template = """
You are a Senior AI Fitness Coach. Your goal is to explain health results in a motivational and scientific way.

--- USER PROFILE ---
- BMI: {bmi}
- Fitness Level: {fitness_level}
- Goal: {goal}
- Food Preference: {food_pref}

--- SCIENTIFIC CONTEXT (RAG) ---
{rag_context}

--- INSTRUCTIONS ---
Provide specific, actionable advice. Avoid generic terms.
1. DIET: Suggest 3 specific meals. E.g., "Grilled Chicken with Quinoa" not just "Chicken".
2. YOGA: 2 specific poses with duration/benefit.
3. WORKOUT: 2 specific exercises (e.g., "30 min brisk walking daily").
4. SUMMARY: A motivational 2-sentence coach note.

Response Format:
DIET: [items]
YOGA: [items]
EXERCISE: [items]
SUMMARY: [text]
"""

prompt = PromptTemplate(
    input_variables=["bmi", "fitness_level", "food_pref", "rag_context"],
    template=coach_prompt_template
)

class CoachService:
    def get_coach_advice(self, bmi, fitness_level, food_pref, goal):
        goal_map = {
            "weight loss": "focusing on a calorie deficit and high-intensity cardio",
            "muscle gain": "prioritizing protein intake and progressive strength training",
            "maintain fitness": "balanced nutrition and consistent moderate activity"
        }
        goal_context = goal_map.get(goal.lower(), "balanced health")

        bmi_reasoning = ""
        if bmi < 18.5:
            advice_type = "Muscle Gain"
            bmi_reasoning = f"Your BMI is {round(bmi, 2)} (Underweight). Since your goal is {goal}, we recommend high-protein meals and weight lifting for healthy growth."
        elif bmi < 25:
            advice_type = "Dynamic Toning"
            bmi_reasoning = f"Your BMI is {round(bmi, 2)} (Healthy). For your goal of {goal}, staying consistent with {goal_context} is key to long-term success."
        else:
            advice_type = "Fat Loss"
            bmi_reasoning = f"Your BMI is {round(bmi, 2)} (Overweight). To achieve {goal}, focus on {goal_context} and increasing your daily step count to 10,000+."

        query = f"User goal is {goal}, BMI {bmi}, fitness {fitness_level}, {food_pref}. Suggest 3 diet items and 2 workout plans."
        rag_hints = rag_instance.retrieve(query, top_k=4)
        rag_context = "\n".join([f"- {fact}" for fact in rag_hints])
        
        if llm:
            try:
                custom_prompt = prompt.format(
                    bmi=round(bmi, 2),
                    fitness_level=fitness_level,
                    food_pref=food_pref,
                    rag_context=rag_context + f"\n- Specific Goal: {goal}\n- Advice Strategy: {advice_type}\n- Reasoning: {bmi_reasoning}"
                )
                # CHANGE: .invoke() instead of .predict() to prevent LangChain 0.2 crash
                response = llm.invoke(custom_prompt)
                response_text = response.content if hasattr(response, "content") else str(response)
                parsed = self._parse_response(response_text)
                parsed["summary"] = f"Goal: {goal.title()}. {bmi_reasoning} {parsed['summary']}"
                return parsed
            except Exception as e:
                print(f"LLM Error: {e}")

        return self._rule_based_fallback(fitness_level, food_pref, bmi, rag_hints, goal)

    def _parse_response(self, text):
        sections = {"diet": "", "yoga": [], "exercise": [], "summary": ""}
        for key in ["DIET", "YOGA", "EXERCISE", "SUMMARY"]:
            match = re.search(f"{key}: (.*?)(?=\n[A-Z]+:|$)", text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if key in ["YOGA", "EXERCISE"]:
                    sections[key.lower()] = [item.strip() for item in content.replace("[", "").replace("]", "").split(",")]
                else:
                    sections[key.lower()] = content
        return sections

    def _rule_based_fallback(self, level, pref, bmi, facts, goal):
        import random
        is_veg = pref.lower() == "veg"
        goal_lower = goal.lower()
        veg_weight_loss = [
            "Lentil Soup + Cucumber Salad", "Quinoa + Roasted Veggies", "Green Tea + Sprout Salad", "Brown Rice + Moong Dal"
        ]
        nonveg_weight_loss = [
            "Grilled Chicken + Broccoli", "Baked Salmon + Asparagus", "Chicken Salad + Olive Oil", "Boiled Eggs + Spinach"
        ]
        veg_muscle = [
            "Tofu Curd + Sweet Potato", "Soy Chunks + Brown Rice", "Paneer Tikka + Green Salad", "Chickpea Stew + Beans"
        ]
        nonveg_muscle = [
            "Chicken Breast + Quinoa", "Steak + Roasted Potatoes", "Eggs + Oatmeal", "Turkey slices + Brown Rice"
        ]
        veg_maintain = [
            "Dal + Brown Rice", "Khichdi with vegetables", "Mixed Veg Curry + Roti", "Oats + Fruit Bowl"
        ]
        nonveg_maintain = [
            "Chicken Curry + Rice", "Fish Tikka + Veggies", "Egg Curry + Roti", "Grilled Chicken wrap"
        ]
        
        veg_underweight = [
            "Energy Shake + Nuts", "Paneer Paratha + Sweet Potato", "Chickpea Curry + Quinoa", "Avocado Toast + Almonds"
        ]
        nonveg_underweight = [
            "Eggs & Bacon + Oatmeal", "Chicken Stew + Brown Rice", "Beef Steak + Mashed Potatoes", "Protein Shake + Tuna Wrap"
        ]
        
        if bmi < 18.5:
            options = veg_underweight if is_veg else nonveg_underweight
            summary = "We need to focus on healthy growth. Ensure you eat energy-dense foods with a caloric surplus!"
            workout = ["Progressive Strength Training", random.choice(["Pushups & Pullups", "Dumbbell exercises"])]
        elif "loss" in goal_lower or bmi > 25:
            options = veg_weight_loss if is_veg else nonveg_weight_loss
            summary = "Let's focus on fat loss and cardiovascular health. Consistency is your superpower."
            workout = ["30 min brisk walking daily", random.choice(["15 min HIIT session", "Stair climbing for 10 min"])]
        elif "gain" in goal_lower:
            options = veg_muscle if is_veg else nonveg_muscle
            summary = "We need to focus on healthy growth. Ensure you eat in a caloric surplus with high protein!"
            workout = ["Progressive Strength Training", random.choice(["Heavy Weightlifting", "Resistance Band exercises"])]
        else:
            options = veg_maintain if is_veg else nonveg_maintain
            summary = "You have a solid foundation! Focus on refining your balance and maintaining this lifestyle."
            workout = ["20 min morning Jogging", random.choice(["Bodyweight Squats (20 reps)", "Core Planks (3 sets)"])]

        diet_list = random.sample(options, min(3, len(options)))
        diet = "\n".join([f"• {item}" for item in diet_list])

        return {
            "diet": diet,
            "yoga": ["Surya Namaskar (10 rounds)", "Balasana (Child Pose)"],
            "exercise": workout,
            "summary": f"Coach says: {summary} Keep following this plan to see measurable results! 🚀"
        }

coach_service = CoachService()
