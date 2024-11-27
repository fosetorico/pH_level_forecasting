import os
import sys

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from dataclasses import dataclass


# @dataclass decorator , because inside any traditional class, to define the class variables you basically use __init__ ,  
# but if we use this @dataclass decorator,  it enables us to define the class variable directly
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    #__init__ since we have other functions to define
    def __init__(self): 
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load dataset
            logging.info("Reading the dataset as a DataFrame")
            df = pd.read_excel('notebook/Dataset/Dataset Disssertation.xlsx')
            logging.info(f"Dataset successfully loaded with shape: {df.shape}")
            
            # Step 1: Remove unsampled rows
            logging.info("Removing unsampled rows...")
            df = df[df['Sample_taken'] == 'Sampled']
            logging.info(f"Dataset after removing unsampled rows: {df.shape}")

            # Step 2: Drop unneeded features
            logging.info("Dropping unneeded features...")
            columns_to_drop = ['WP_ID', 'DataType', 'Date_Assessment_Original', 'SURVEY_DETAIL_ID',
                            'COUNTRY', 'Comment', 'HCO3', 'Corrected_HCO3', 'Sample_taken', 
                            'Date_Assessment', 'Time_Assessment']
            df = df.drop(columns=columns_to_drop, axis=1)
            logging.info(f"Remaining columns after dropping: {df.columns.tolist()}")

            # Step 3: Impute missing values
            logging.info("Imputing missing values...")
            imputer = SimpleImputer(strategy='median')
            columns_to_impute = ['Total Iron (mg/l)', 'Tryptophan_Probe_µgL']
            df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
            logging.info(f"Missing values imputed for columns: {columns_to_impute}")

            # Step 4: Outlier detection and restoration
            logging.info("Performing outlier detection and restoration...")
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            for column in df.columns:
                outliers = (df[column] < lower_bound[column]) | (df[column] > upper_bound[column])
                if outliers.any():
                    median_value = df[column].median()
                    df.loc[outliers, column] = median_value
            logging.info("Outlier detection and replacement completed.")

            # Save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved.")

            # Step 5: Train-test split
            logging.info("Initiating train-test split...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train-test split completed.")

            logging.info("Ingestion of the data is complete")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Exception occurred during data ingestion.")
            raise CustomException(e, sys)
        

if __name__=="__main__":
    obj=DataIngestion()     
    obj.initiate_data_ingestion() 
    # train_data,test_data=obj.initiate_data_ingestion()

    # data_transformation=DataTransformation()
    # data_transformation.initiate_data_transformation(train_data,test_data)
    # train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)    

    # modeltrainer=ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))