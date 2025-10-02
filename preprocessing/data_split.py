from typing import Union
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod 
import pandas as pd


class DataSplit(ABC):
    """
    Abstract class for data splitting
    
    """

    @abstractmethod
    def split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Abstractmethod for Split the data
        
        Args:
            data (pd.DataFrame): the input data to split on
        """
        pass


class DataSplit(DataSplit):
    """Class for data splitting"""

    def split_data(self, data: pd.DataFrame,
                   test_size: float = 0.15) -> Union[pd.DataFrame, pd.Series]:
        """
        Split the data into train validation and test sets
        
        Args:
            data (pd.DataFrame): The input data ingested for splitting
            test_size (float): The size of the test set. Default is 0.15
            
        Returns:
            Union[pd.DatFrame, pd.Series]: The train, validation and test splits 
            of both label and target
            
            """

        try:

            X = data.drop(columns='Churn', axis=1)
            y = (data['Churn']=='Yes').astype(int)
            
            test_size = 0.15
            val_ratio = test_size / (1 - test_size)

            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
                )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
            )

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            print("Splitting into data sets failed:", e)
            raise e