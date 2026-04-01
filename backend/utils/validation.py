def validate_fitness_input(age, weight, height):
    if age <= 0 or age > 110:
        raise ValueError("Age must be a positive value between 1 and 110.")
    if height <= 0:
        raise ValueError("Height must be greater than 0.")
    if weight <= 0:
        raise ValueError("Weight must be greater than 0.")
    if height > 3.0:
        raise ValueError("Height seems unrealistic (> 3m). Please check.")
    if weight > 500:
        raise ValueError("Weight seems unrealistic (> 500kg). Please check.")


def validate_heart_input(bp, cholesterol):
    if not (50 <= bp <= 250):
        raise ValueError(
            "Resting Blood Pressure must be in a realistic range (50-250)."
        )
    if not (80 <= cholesterol <= 600):
        raise ValueError("Cholesterol must be in a realistic range (80-600).")
