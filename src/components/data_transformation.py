import sys
from dataclasses import dataclass
import os

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.exception import CustomException
from src.logger import logging
from src.utils import OutlierHandler, save_object

# @dataclass decorator , because inside any traditional class, to define the class variables you basically use _init_ ,  
# but if we use this @dataclass decorator,  it enables us to define the class variable directly
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns):
        """
        This function creates and returns a data preprocessing pipeline,
        including missing value imputation, scaling, and outlier handling.
        """
        try:
            # Define the pipeline
            # numerical_columns = ['Temp (oC)','SEC (µS/cm)', 'Turbidity (<NTU)', 'Total Iron (mg/l)', 'Titration 1', 'Titration 2', 'Volume 50/100ml', 'N_VALUE', 'Tryptophan_Probe_µgL', 'Final HCO3']
            num_pipeline = Pipeline(
                steps=[ 
                    ("imputer", SimpleImputer(strategy="median")),
                    ("outlier_handler", OutlierHandler()),  
                    ("scaler", StandardScaler()),  
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Removing unsampled rows...")
            train_df = train_df[train_df['Sample_taken'] == 'Sampled']
            test_df = test_df[test_df['Sample_taken'] == 'Sampled']
            logging.info(f"Train dataset shape after removing unsampled rows: {train_df.shape}")
            logging.info(f"Test dataset shape after removing unsampled rows: {test_df.shape}")

            logging.info("Dropping unneeded features...")
            columns_to_drop = ['WP_ID', 'DataType', 'Date_Assessment_Original', 'SURVEY_DETAIL_ID',
                               'COUNTRY', 'Comment', 'HCO3', 'Corrected_HCO3', 'Sample_taken', 
                               'Date_Assessment', 'Time_Assessment']
            train_df = train_df.drop(columns=columns_to_drop, axis=1)
            test_df = test_df.drop(columns=columns_to_drop, axis=1)
            logging.info(f"Remaining columns: {train_df.columns.tolist()}")   

            logging.info("Creating preprocessing pipeline...")
            numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
            target_column_name = 'pH'  # Replace with your actual target column name
            numerical_columns.remove(target_column_name)

            preprocessing_obj = self.get_data_transformer_object(numerical_columns)

            logging.info(f"Applying preprocessing object on training and testing datasets.")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)

            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error("Exception occurred during data transformation.")
            raise CustomException(e, sys)        