import os
import sys
import re

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def extract_base_model(name: str) -> str:
    """
    Cleans a raw vehicle model name and returns the base model only.
    No encoding is done here — just cleaning for consistency before encoding.
    """
    if pd.isnull(name) or name.strip() == "":
        return "unknown"

    name = name.lower()
    name = re.sub(r'[^a-z0-9 ]', '', name)  # removing special characters
    name = re.sub(
        r'\b(crew|cab|pickup|sedan|coupe|van|wagon|truck|convertible|utility|hatchback|2d|4d|4x4|fx4|awd|fwd|rwd|sr|ex|lx|le|lt|xlt|sel|slt|premium|limited|base|plus|l|gls|xle|se|xl|sport|touring|super|luxury|classic|series|class)\b',
        '', name
    )
    name = re.sub(r'\s+', ' ', name).strip()
    
    # returning just the first word (base model)
    return name.split()[0] if name else "unknown"