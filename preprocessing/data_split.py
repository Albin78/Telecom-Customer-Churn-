from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from abc import ABC, abstractmethod 
import numpy as np
import pandas as pd


class DataSplit(ABC):
    """Abstract class for data splitting"""

    @abstractmethod
    def split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data"""
        pass


class DataSplit(DataSplit):
    """Class for data splitting"""

    def split_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """Split the data inot train validation and test sets"""

        try:

            X = data.drop(columns=['Churn', 'customerID', 'gender', 'PhoneService'])
            y = (data['Churn']=='Yes').astype(int)

            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.20, stratify=y, random_state=42
                )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
            )

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            print("Splitting into data sets failed:", e)
            raise e