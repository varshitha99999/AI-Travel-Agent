def get_weather(destination: str) -> str:
    """Get weather information for popular Indian destinations"""
    
    # Weather data for popular destinations (seasonal patterns)
    weather_db = {
        "goa": {
            "best_time": "November to February",
            "current": "Warm and humid, 28-32°C",
            "tip": "Carry sunscreen and light cotton clothes"
        },
        "manali": {
            "best_time": "March to June, December to February (for snow)",
            "current": "Cool to cold, 10-20°C",
            "tip": "Pack warm clothes and jackets"
        },
        "jaipur": {
            "best_time": "October to March",
            "current": "Pleasant to warm, 20-30°C",
            "tip": "Carry sunglasses and stay hydrated"
        },
        "kerala": {
            "best_time": "September to March",
            "current": "Tropical and humid, 25-32°C",
            "tip": "Light clothes and rain gear recommended"
        },
        "delhi": {
            "best_time": "October to March",
            "current": "Variable, 15-35°C depending on season",
            "tip": "Check seasonal weather before packing"
        },
        "mumbai": {
            "best_time": "November to February",
            "current": "Humid and warm, 25-33°C",
            "tip": "Light breathable fabrics work best"
        },
        "udaipur": {
            "best_time": "September to March",
            "current": "Pleasant, 18-28°C",
            "tip": "Comfortable weather for sightseeing"
        },
        "shimla": {
            "best_time": "March to June, December to February (for snow)",
            "current": "Cool to cold, 8-18°C",
            "tip": "Warm clothes essential, especially evenings"
        },
    }
    
    # Normalize destination name
    dest_lower = destination.lower().strip()
    
    # Find weather info
    weather = weather_db.get(dest_lower)
    
    if weather:
        return f"""🌤️ **Weather in {destination.title()}:**

📅 **Best Time to Visit:** {weather['best_time']}
🌡️ **Current Conditions:** {weather['current']}
💡 **Travel Tip:** {weather['tip']}

Note: Weather can vary, check forecast closer to your travel date!"""
    else:
        # Generic response for unlisted destinations
        return f"""🌤️ **Weather in {destination.title()}:**

For accurate weather information, check:
• weather.com
• AccuWeather
• India Meteorological Department (IMD)

💡 General India Travel Tips:
- Summer (Mar-Jun): Hot, 30-45°C
- Monsoon (Jul-Sep): Rainy, 25-35°C
- Winter (Oct-Feb): Pleasant, 10-25°C"""