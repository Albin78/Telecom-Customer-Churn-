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
            probs = self.model.predict_proba(X, num_iteration=self._load_iterator())[:, 1]
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
            cat_cols = input.select_dtypes(exclude=['number']).columns.tolist()

            for c in cat_cols:
                input[c] = input[c].astype("category")

            preds, probs = self.predict(input)

            return preds, probs

        except Exception as e:
            print("Error occured during prediction from input dict as :", str(e))
            raise e




def main():

    pipeline = LGBMPipeline(save_path="models_artifact/")
    average_precision, roc_auc, clf_report, conf_matrx = pipeline.estimator()

    print("\n\nAverage Precision:", average_precision)
    print("\nROC-AUC:", roc_auc)
    print("\nClassification Report:\n\n", clf_report)
    print("\nConfusion Matrix:\n", conf_matrx)

    predictor = LGBMPredictor(artificats_dir="models_artifact/")

    # model = predictor._load_model()
    # print("\n\nModel Features:", model.feature_name_)
    
    input = {
        'SeniorCitizen': 0,
        'Partner': 1,
        'Dependents':0,
        'tenure':12,
        'InternetService':"DSL",
        "Contract": "One year",
        "PaperlessBilling": 0,
        "PaymentMethod": "Electronic check",
        'MonthlyCharges': 20.5,
        'TotalCharges':430.7,
        'Fibre_stream_pref':0,
        'DSL_security_pref': 1,
        'has_phone': 1,
        'has_multipleline': 0,
        "tenure_bucket": "4-12",
        "avg_monthly_charge": 430.7 / 12,    
        'total_addons': 1,
        "contract_payment": "One year_Electronic check",
        "security_bins": 1,
        'streaming_bins': 2,
        'new_customers': 0,
        'spend_per_addon': 20.12
        }

    # ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'InternetService', 'Contract',
    #  'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Fibre_stream_pref', 
    #  'DSL_security_pref', 'has_phone', 'has_multipleline', 'tenure_bucket', 'avg_monthly_charge', 
    #  'total_addons', 'contract_payment', 'security_bins', 'streaming_bins', 'new_customers', 'spend_per_addon']

    # avg_monthly_charge = input["MonthlyCharges"] / input["tenure"]

    # input["avg_monthly_charge"] = avg_monthly_charge

    preds, probs = predictor.predict_from_dict(input)

    print("Prediction Label:", preds)
    print("\nPrediction Probabilities:", probs)


if __name__ == "__main__":
    main()
