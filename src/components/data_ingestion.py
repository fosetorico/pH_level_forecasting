import os
import sys

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

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

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_excel('notebook/Dataset/Dataset Disssertation.xlsx')
            logging.info('Reading the dataset as dataframe')
            logging.info(df.columns)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            ) 
              
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()     
    # obj.initiate_data_ingestion() 
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    # data_transformation.initiate_data_transformation(train_data,test_data)
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)    

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))