class WeatherService:
    def get_forecast(self, location, dates):
        return {"location": location, "dates": dates, "forecast": []}
