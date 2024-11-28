import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin


# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.exception import CustomException


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle outliers by replacing them with the median.
    """
    def __init__(self, factor=1.5):
        self.factor = factor
        self.feature_names = None

    def fit(self, X, y=None):
        # Convert to DataFrame if input is NumPy array
        if isinstance(X, np.ndarray):
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names)
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            raise ValueError("Input data must be a DataFrame or a NumPy array.")

        # Calculate IQR, lower, and upper bounds for each column
        self.Q1 = X.quantile(0.25)
        self.Q3 = X.quantile(0.75)
        self.IQR = self.Q3 - self.Q1
        self.lower_bound = self.Q1 - self.factor * self.IQR
        self.upper_bound = self.Q3 + self.factor * self.IQR
        return self

    def transform(self, X, y=None):
        # Convert to DataFrame if input is NumPy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a DataFrame or a NumPy array.")

        X = X.copy()
        for column in X.columns:
            # Replace outliers with median
            outliers = (X[column] < self.lower_bound[column]) | (X[column] > self.upper_bound[column])
            if outliers.any():
                median_value = X[column].median()
                X.loc[outliers, column] = median_value

        # Convert back to NumPy array for pipeline compatibility
        return X.to_numpy()


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
        print(f"Loading object from: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)