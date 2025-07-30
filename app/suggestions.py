def get_suggestion(stress_level):
    if stress_level == "Low":
        return "Keep it up! Stay balanced with regular breaks."
    elif stress_level == "Medium":
        return "Take a short walk or do some breathing exercises."
    elif stress_level == "High":
        return "Consider talking to a friend or using a mindfulness app."
    else:
        return "Stress level not recognized."
