import unittest
import pandas as pd
import joblib
import numpy as np
from weather_api import fetch_weather

class TestWearSmartModules(unittest.TestCase):
    
    def setUp(self):
        self.api_key = "4c703f15e3f9220de836884137342d5d"  # Replace with your actual key if needed
        self.test_city = "Lahore"
        self.model = joblib.load("weather_clothing_recommender.pkl")
        self.clothing_df = pd.read_csv("enhanced_weather_clothing_dataset_no_accessories.csv")
        self.clothing_df.columns = self.clothing_df.columns.str.strip().str.lower()

    def test_fetch_weather(self):
        """Test if weather API returns the expected dictionary structure."""
        weather = fetch_weather(self.test_city, self.api_key)
        self.assertIsInstance(weather, dict)
        self.assertIn("temperature", weather)
        self.assertIn("humidity", weather)
        self.assertIn("wind", weather)
        self.assertIn("condition", weather)
        self.assertIn("description", weather)

    def test_model_prediction_structure(self):
        """Test if model returns a 3-part clothing prediction."""
        dummy_input = {
            'temperature': 25,
            'feels_like': 25,
            'humidity': 50,
            'wind_speed': 5,
            'weather_condition': "clear",
            'time_of_day': "morning",
            'season': "summer",
            'mood': "good",
            'occasion': "casual"
        }
        input_df = pd.DataFrame([dummy_input])
        prediction = self.model.predict(input_df)[0]
        self.assertIsInstance(prediction, (list, tuple, np.ndarray))
        self.assertEqual(len(prediction), 3)

    def test_clothing_filtering_logic(self):
        """Test if filtered clothing DataFrame returns valid items."""
        prediction = ("shirt", "jeans", "none")  # example
        filtered = self.clothing_df[
            (self.clothing_df["recommended_top"] == prediction[0]) &
            (self.clothing_df["recommended_bottom"] == prediction[1]) &
            (self.clothing_df["recommended_outer"] == prediction[2])
        ]
        self.assertIsInstance(filtered, pd.DataFrame)

if __name__ == "__main__":
    unittest.main()
