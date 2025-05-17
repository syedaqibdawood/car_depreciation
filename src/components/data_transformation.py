import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, extract_base_model

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
            all_categorical_features = [
                "manufacturer", "fuel", "title_status", "model",
                "condition", "cylinders", "type", "transmission"
            ]
            mode_fill_columns = ["transmission"]
            general_cats = [col for col in all_categorical_features if col not in mode_fill_columns]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline_unknown = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ("scaler", StandardScaler(with_mean=False))
            ])

            cat_pipeline_mode = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline_unknown", cat_pipeline_unknown, general_cats),
                ("cat_pipeline_mode", cat_pipeline_mode, mode_fill_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def remove_outliers_iqr(self, df, cols):
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading training and testing datasets...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Dropping unnecessary columns
            cols_to_drop = ["county", "size", "state", "paint_color", "region", "posting_date", "drive"]
            train_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            test_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

            # Dropping rows with missing critical values
            train_df.dropna(subset=["year", "odometer", "fuel", "model"], inplace=True)
            test_df.dropna(subset=["year", "odometer", "fuel", "model"], inplace=True)

            # Feature engineering: car_age
            current_year = 2025
            train_df["car_age"] = current_year - train_df["year"]
            test_df["car_age"] = current_year - test_df["year"]
            train_df.drop(columns=["year"], inplace=True)
            test_df.drop(columns=["year"], inplace=True)

            # Model column cleaning
            train_df["model"] = train_df["model"].apply(extract_base_model)
            test_df["model"] = test_df["model"].apply(extract_base_model)

            # Converting cylinders to numeric
            for df_ in [train_df, test_df]:
                if df_['cylinders'].dtype == 'object':
                    df_['cylinders'] = df_['cylinders'].str.extract(r'(\d+)')
                    df_['cylinders'] = pd.to_numeric(df_['cylinders'], errors='coerce')

            # Dropping rows with missing numerical columns
            train_df.dropna(subset=['price', 'odometer', 'car_age'], inplace=True)
            test_df.dropna(subset=['price', 'odometer', 'car_age'], inplace=True)

            # Removing outliers
            num_cols = ['price', 'odometer', 'car_age']
            train_df = self.remove_outliers_iqr(train_df, num_cols)

            # Converting categorical columns to string
            cat_cols = ["manufacturer", "fuel", "title_status", "model", "condition", "cylinders", "type", "transmission"]
            for col in cat_cols:
                train_df[col] = train_df[col].astype(str)
                test_df[col] = test_df[col].astype(str)

            # Log-transforming the target column
            train_df['price'] = np.log1p(train_df['price'])
            test_df['price'] = np.log1p(test_df['price'])

            target_column = "price"
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            preprocessor = self.get_data_transformer_object()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Data transformation complete and preprocessor saved.")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
