def calculate_budget(input_text: str) -> str:
    try:
        parts = input_text.split(",")
        total_budget = float(parts[0])
        days = int(parts[1])

        per_day = total_budget / days
        return f"Recommended daily budget: {per_day:.2f} INR"
    except:
        return "Provide input as: total_budget,days"