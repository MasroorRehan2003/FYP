import requests

def fetch_weather(city, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()

        return {
            "city": data.get("name"),
            "temperature": data["main"].get("temp"),
            "humidity": data["main"].get("humidity"),        # ✅ FIXED
            "wind": data["wind"].get("speed"),               # ✅ FIXED
            "condition": data["weather"][0].get("main"),
            "description": data["weather"][0].get("description")
        }
    except Exception as e:
        print("Error fetching weather:", e)
        return None
