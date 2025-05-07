import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.utils import load_object, extract_base_model
from src.exception import CustomException

@dataclass
class CustomData:
    manufacturer: str
    model: str
    title_status: str
    fuel: str 
    year: int
    odometer: float

    def get_data_as_data_frame(self, car_age, updated_odometer):
        try:
            # Cleaning model value
            cleaned_model = extract_base_model(self.model)

            custom_data_input_dict = {
                "manufacturer": [self.manufacturer],
                "model": [cleaned_model],
                "title_status": [self.title_status],
                "fuel": [self.fuel], 
                "car_age": [car_age],
                "odometer": [updated_odometer]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.future_years = 4  # Predicting Price for 2025 to 2028

    def predict(self, data: CustomData):
        try:
            print("Before Loading")
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            print("After Loading")

            base_year = 2025
            current_age = base_year - data.year
            current_odometer = data.odometer

            # Estimating miles per year
            if current_age > 0:
                avg_miles_per_year = current_odometer / current_age
            else:
                avg_miles_per_year = 12000  # fallback for new cars

            results = []

            for i in range(self.future_years):
                car_age = current_age + i
                odometer_future = current_odometer + (i * avg_miles_per_year)

                input_df = data.get_data_as_data_frame(
                    car_age=car_age,
                    updated_odometer=odometer_future
                )

                print(f"\n Predicting for year {base_year + i} | Age: {car_age} | Odo: {odometer_future}")
                input_scaled = preprocessor.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                results.append(round(prediction, 2))

            return results

        except Exception as e:
            raise CustomException(e, sys)