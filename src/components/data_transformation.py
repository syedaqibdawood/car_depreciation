import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Setting up preprocessing pipelines...")

            numerical_features = ["odometer", "car_age"]
            categorical_features = [
                "manufacturer", "model", "title_status"
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading training and testing datasets...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Step 1: Dropping unwanted columns
            cols_to_drop = [
                "county", "size", "state", "paint_color", "type", 
                "region", "condition", "cylinders", "drive", 
                "fuel", "transmission", "posting_date"
            ]
            train_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            test_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

            # Step 2: Dropping rows with missing critical values
            train_df.dropna(subset=["year", "odometer", "model"], inplace=True)
            test_df.dropna(subset=["year", "odometer", "model"], inplace=True)

            # Step 3: Feature engineering - car_age
            current_year = 2025
            train_df["car_age"] = current_year - train_df["year"]
            test_df["car_age"] = current_year - test_df["year"]
            train_df.drop(columns=["year"], inplace=True)
            test_df.drop(columns=["year"], inplace=True)

            # Step 4: Separating features and target
            target_column = "price"
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Step 5: Transforming data
            preprocessor = self.get_data_transformer_object()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Step 6: Combining transformed features with target
            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            # Step 7: Saving preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Data transformation completed and preprocessor saved.")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)