from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipe
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


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


    def preprocess(self):

        """
        Handles preprocessing pipelines and composing into a single Transformer

        
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
                                    

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numeric_features),
                    ('cat', cat_bin_transformer, self.cat_bin_features)
                    ])
            
            return preprocessor
        
        except Exception as e:
            print("Error happend while preprocessing:", e)
            raise e


    def model_pipeline(self, model_type: str,
                      smote: bool=False):

        """
        The model pipeline for LR and Tree for fitting
        
        Args:
            model_type (str): The specified model for building pipeline
            smote (bool): The SMOTE method for class imbalance
        
        Returns: Pipeline

        """
        
        try:

            preprocessor = self.preprocess()

            if model_type == 'lr':

                estimator = LogisticRegression(solver="saga", penalty="l2", max_iter=1000,
                                               random_state=42, class_weight='balanced')
                

            elif model_type == 'lr_unbalanced':

                estimator = LogisticRegression(solver="saga", penalty="l2", max_iter=1000,
                                               random_state=42)

            elif model_type == 'rf':

                estimator = RandomForestClassifier(random_state=42, class_weight='balanced')

            elif model_type == 'rf_unbalanced':

                estimator = RandomForestClassifier(random_state=42)

            
            if smote:

                pipeline = ImbPipe([
                    ("preprocessor", preprocessor),
                    ("smote", SMOTE(random_state=42)),
                    ("model", estimator)
                ])

            else:
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", estimator)
                ])
                
            return pipeline
        
        except Exception as e:
            print("Error occured while processing model pipeline as :", str(e))
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