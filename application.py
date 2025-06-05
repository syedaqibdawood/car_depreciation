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
            # Collect form data
            form_values = {
                "manufacturer": request.form.get('manufacturer'),
                "model": request.form.get('model'),
                "title_status": request.form.get('title_status'),
                "fuel": request.form.get('fuel'), 
                "year": request.form.get('year'),
                "odometer": request.form.get('odometer'),
                "condition": request.form.get('condition'),
                "cylinders": request.form.get('cylinders'),
                "type": request.form.get('type'),
                "transmission": request.form.get('transmission')
            }

            # Prepare input for pipeline
            input_data = CustomData(
                manufacturer=form_values['manufacturer'],
                model=form_values['model'],
                title_status=form_values['title_status'],
                fuel=form_values['fuel'], 
                year=int(form_values['year']),
                odometer=float(form_values['odometer']),
                condition=form_values['condition'],
                cylinders=form_values['cylinders'],
                type=form_values['type'],
                transmission=form_values['transmission']
            )

            # Predict
            pipeline = PredictPipeline()
            results = pipeline.predict(input_data)

            # Extract 2025 prediction only
            predicted_2025 = results[0] if results and len(results) > 0 else None
            predictions = {"2025": predicted_2025}

            return render_template('home.html', predictions=predictions, form_values=form_values)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
