import pandas as pd
import numpy as np
import json
import joblib
import os


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
            raise FileNotFoundError(f"The iterator file path {iterator_path} was found")

        with open(iterator_path, "r") as f:
            return json.load(f)["best iterator"]

    
    def predict(
        self, X: pd.DataFrame
        ):

        """
        Predict the input dataframe using the
        loaded model.
        
        Args:
            X (pd.DataFrame): The input dataframe

        Returns:
            preds, probs (tuple(int, float)): The prediction label 
                                              and prediction probabilities

        """
        
        try:
            probs = self.model.predict_proba(X, n_estimators=self._load_iterator())[:, 1]
            preds = (probs >= self._load_threshold())

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

            input = pd.DataFrame([input_dict])
            preds, probs = self.predict(input)

            return preds, probs

        except Exception as e:
            print("Error occured during prediction from input dict as :", str(e))
            raise e


