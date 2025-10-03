import pandas as pd
import numpy as np
import json
import joblib
import os
from predict.pipeline import LGBMPipeline


class LGBMPredictor:

    def __init__(
        self, artificats_dir:str
        ):

        """Predictor for the inputs using LGBM
        model.
        
        Args:
            artificats_dir (str): The file path where
                                  the artifacts are stored
    
        """

        self.artifacts_dir = artificats_dir
        self.model = self._load_model()
        self.threshold = self._load_threshold()
        self.best_iteration = self._load_iterator()
        self.train_metadata = self._load_metadata()

    
    def _load_model(self):

        """
        Load the saved model from the artifacts stored file
        
        """

        model_path = os.path.join(self.artifacts_dir, "final_estimator.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found in the file path: {model_path}")
        
        return joblib.load(model_path)

    
    def _load_threshold(self):

        """
        Load the threshold which is saved inside
        the artifacts directory
        
        """

        threshold_path = os.path.join(self.artifacts_dir, "final_threshold.json")

        if not os.path.exists(threshold_path):
            raise FileNotFoundError(f"Threshold not found in the file path: {threshold_path}")
        
        with open(threshold_path, "r") as f:
            return json.load(f)["Threshhold"]

    
    def _load_iterator(self):

        """
        Loads the best iterator which was saved on artifacts
        that is got from the training
        
        """

        iterator_path = os.path.join(self.artifacts_dir, "estimator_metadata.json")

        if not os.path.exists(iterator_path):
            raise FileNotFoundError(f"The iterator file path {iterator_path} not found")

        with open(iterator_path, "r") as f:
            return json.load(f)["best iterator"]
    

    def _load_metadata(self):

        """
        Load the training metadata
        
        """

        metadata_path = os.path.join(self.artifacts_dir, "training_meta.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"The metadata path {metadata_path} not found")

        with open(metadata_path, "r") as f:
            return json.load(f)

    
    def predict(
        self, input_df: pd.DataFrame
        ):

        """
        Predict the input dataframe using the
        loaded model.
        
        Args:
            input (pd.DataFrame): The input dataframe

        Returns:
            preds, probs (tuple(int, float)): The prediction label 
                                              and prediction probabilities

        """
        
        try:

            input_df = input_df.reindex(columns=self.train_metadata["features"], fill_value=np.nan)

            probs = self.model.predict_proba(input_df, num_iteration=self._load_iterator())[:, 1]
            preds = (probs >= self._load_threshold()).astype(int)

        except Exception as e:
            print("Error occured during final prediction as:", str(e))
            raise e

        return preds, probs
    

    def predict_from_dict(
        self, input_dict: dict
        ):

        """
        Predict the probabilities and labels from the 
        input dictionary
        
        Args:
            input_dict (dict): Input dictionary with the input parameters
                               for predictions
        
        Returns:
            tuple(int, float): prediction label, predicted probabilities

        """

        try:

            input_df = pd.DataFrame([input_dict])

            input_df = input_df.reindex(columns=self.train_metadata["features"], fill_value=np.nan)

            for col, categories in self.train_metadata.get("categorical_mappings", {}).items():

                if col in input_df.columns:
                    input_df[col] = pd.Categorical(input_df[col], categories=categories)
            
            for col, dtype in self.train_metadata.get("dtypes", {}).items():

                if col in input_df.columns:
                    if dtype.startswith("int") or dtype.startswith("float"):
                        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
            

            preds, probs = self.predict(input_df)

            return preds, probs

        except Exception as e:
            print("Error occured during prediction from input dict as :", str(e))
            raise e
