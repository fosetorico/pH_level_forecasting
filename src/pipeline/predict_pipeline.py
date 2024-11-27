import sys
import pandas as pd
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline: 
    def __init__(self):
        pass


    # model prediction pipe
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            print("Features before preprocessing:")
            print(features)            
            data_scaled=preprocessor.transform(features)
            print("Features after scaling:")
            print(data_scaled)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        

# mapping the input in the HTML to the backend
class CustomData:
    def __init__(self,
        Temp: float,
        SEC: float,
        Turbidity:float,
        Total_Iron: float,
        Titration_1: float,
        Titration_2: float,
        Volume,
        N_VALUE: float,
        Tryptophan_Probe: float,
        Final_HCO3: float,
        ):

        self.Temp = Temp
        self.SEC = SEC
        self.Turbidity = Turbidity
        self.Total_Iron = Total_Iron
        self.Titration_1 = Titration_1
        self.Titration_2 = Titration_2
        self.Volume = Volume       
        self.N_VALUE = N_VALUE       
        self.Tryptophan_Probe = Tryptophan_Probe       
        self.Final_HCO3 = Final_HCO3       

    # return all input in form of a dataframe
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Temp": [self.Temp],
                "SEC": [self.SEC],
                "Turbidity": [self.Turbidity],
                "Total_Iron": [self.Total_Iron],
                "Titration_1": [self.Titration_1],
                "Titration_2": [self.Titration_2],
                "Volume": [self.Volume],
                "N_VALUE": [self.N_VALUE],
                "Tryptophan_Probe": [self.Tryptophan_Probe],
                "Final_HCO3": [self.Final_HCO3],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys) 