from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class PreprocessSetup:
    """Setup the preprocessing pipeline for the  data"""

    def __init__(self, data: pd.DataFrame, numeric_features: list[str],
                cat_bin_features: list[str]):

        """
        Initialize the PreprocessSetup class
        
        Args:
            data: pd.DataFrame
            numeric_features: list[str]
            cat_bin_features: list[str]
        
        Returns:
            None
        """

        self.data = data
        self.numeric_features = numeric_features
        self.cat_bin_features = cat_bin_features
        self.preprocessor = None

    def preprocess(self, model_type: str):

        """
        Handles preprocessing pipelines and composing into a single Transformer

        Args:
            model_type (str): The type of model to preprocess for
        
        Returns:
            self.preprocessor (ColumnTransformer): The preprocessor pipeline
        """

        try:

            numeric_transformer = Pipeline(
                steps=[
                    ('scalar', StandardScaler())
                ])
            
            cat_bin_transformer = Pipeline(
                steps=[
                    ('OHE', OneHotEncoder(drop='first', handle_unknown='ignore'))
                ])
                                    
            
            if model_type == "lr":

                self.preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, self.numeric_features),
                        ('cat', cat_bin_transformer, self.cat_bin_features)
                    ])

            elif model_type == "tree":

                self.preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', 'passthrough', self.numeric_features),
                        ('cat', cat_bin_transformer, self.cat_bin_features)
                    ])
            
            return self.preprocessor
        
        except Exception as e:
            print("Error happend while preprocessing:", e)
            raise e
    
    def fit_transform(self, split: pd.DataFrame):

        """
        Fit and transform the data split using preprocessor
        
        Args: 
            split (pd.DataFrame): The data split to fit and transform

        Returns:
            np.ndarray: The transformed data
        """
        
        return self.preprocessor.fit_transform(split)

    def transform(self, split: pd.DataFrame):

        """
        Transform the data split using preprocessor

        Args:
            split (pd.DataFrame): The data split to transform

        Returns:
            np.ndarray: The transformed data

        """

        return self.preprocessor.transform(split)