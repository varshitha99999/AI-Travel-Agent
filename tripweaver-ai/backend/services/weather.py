import requests
from typing import Dict, Optional

USER_AGENT = "TripWeaverAI/1.0 (contact: support@tripweaver.ai)"

def _geocode_city(city: str) -> Optional[Dict]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]), "display_name": data[0]["display_name"]}

def _weather_code_to_text(code: int) -> str:
    mapping = {
        0: "Clear",
        1: "Mainly Clear",
        2: "Partly Cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing Rime Fog",
        51: "Light Drizzle",
        53: "Moderate Drizzle",
        55: "Dense Drizzle",
        61: "Slight Rain",
        63: "Moderate Rain",
        65: "Heavy Rain",
        71: "Slight Snowfall",
        73: "Moderate Snowfall",
        75: "Heavy Snowfall",
        80: "Rain Showers",
        81: "Moderate Rain Showers",
        82: "Violent Rain Showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return mapping.get(code, "Unknown")

def get_weather(city: str) -> str:
    # Read env at call time to ensure .env has been loaded
    import os
    weather_provider = os.getenv("WEATHER_PROVIDER")
    weatherstack_api_key = os.getenv("WEATHERSTACK_API_KEY") or os.getenv("WEATHER_API_KEY")

    if (weather_provider and weather_provider.lower() == "weatherstack") or weatherstack_api_key:
        try:
            url = "http://api.weatherstack.com/current"
            params = {"access_key": weatherstack_api_key, "query": city}
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if data.get("error"):
                return f"❌ Weather service error for {city.title()}: {data['error'].get('info','Unknown error')}"
            loc = data.get("location", {})
            cur = data.get("current", {})
            name = loc.get("name") or city
            temp = cur.get("temperature")
            humidity = cur.get("humidity")
            descs = cur.get("weather_descriptions") or []
            cond = descs[0] if descs else "Unknown"
            return (
                f"🌤 Weather in {str(name)}\n\n"
                f"Temperature: {temp}°C\n"
                f"Condition: {cond}\n"
                f"Humidity: {humidity}%\n"
                f"Source: Weatherstack"
            )
        except requests.HTTPError as e:
            return f"❌ Weather service error for {city.title()}: {e.response.status_code}"
        except Exception as e:
            return f"❌ Could not fetch weather for {city.title()}: {str(e)}"
    try:
        loc = _geocode_city(city)
        if not loc:
            return f"🌤 Weather in {city.title()}\n\nCould not locate this city."
        lat, lon = loc["lat"], loc["lon"]
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code",
            "timezone": "auto",
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        current = data.get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        code = current.get("weather_code")
        condition = _weather_code_to_text(code) if code is not None else "Unknown"
        return (
            f"🌤 Weather in {city.title()}\n\n"
            f"Temperature: {temp}°C\n"
            f"Condition: {condition}\n"
            f"Humidity: {humidity}%\n"
            f"Source: Open‑Meteo"
        )
    except requests.HTTPError as e:
        return f"❌ Weather service error for {city.title()}: {e.response.status_code}"
    except Exception as e:
        return f"❌ Could not fetch weather for {city.title()}: {str(e)}"
