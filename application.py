import sys
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collecting form data from HTML
            manufacturer = request.form.get('manufacturer')
            model = request.form.get('model')
            title_status = request.form.get('title_status')
            fuel = request.form.get('fuel') 
            year = int(request.form.get('year'))
            odometer = float(request.form.get('odometer'))
            condition = request.form.get('condition')
            cylinders = request.form.get('cylinders')
            car_type = request.form.get('type')
            transmission = request.form.get('transmission')

            # Packaging input into CustomData
            input_data = CustomData(
                manufacturer=manufacturer,
                model=model,
                title_status=title_status,
                fuel=fuel, 
                year=year,
                odometer=odometer,
                condition=condition,
                cylinders=cylinders,
                type=car_type,
                transmission=transmission
            )

            # Predicting using PredictPipeline
            print("Sending data to prediction pipeline...")
            pipeline = PredictPipeline()
            results = pipeline.predict(input_data)

            # Showing all results — price for 2025, 2026, 2027, 2028
            future_years = [2025 + i for i in range(len(results))]
            predictions = dict(zip(future_years, results))

            print("Prediction complete:", predictions)

            return render_template('home.html', predictions=predictions)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
