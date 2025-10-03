from typing import Tuple
from preprocessing.data_split import DataSplit
from preprocessing.ingest_clean import IngestData, CleanData
from models.LGBM import LGBMModel
from sklearn.model_selection import StratifiedKFold
from models.metrics import final_model_prediction
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import joblib
import json


class LGBMPipeline:

    def __init__(self, save_path: str):
        
        self.save_path = save_path

    def run_pipeline(self):

        """
        LGBM Model pipeline with ingesting data, cleaning
        and fitting on folds. 
        
        Returns:
            tuple(list, dict, list, list): The best iterations list,
                                           parameters that fitted well,
                                           out of fold predictions, list
                                           of models
            
        """
        
        try:
            ingest_data = IngestData(file_path=r"dataset\feature_engineered_data.csv").get_data()
            data = CleanData(data=ingest_data).feature_engineer()

            categorical_features = data.select_dtypes(exclude=['number']).columns.tolist()

            X, y, X_train, X_val, X_test, y_train, y_val, y_test = DataSplit().split_data(data=data, test_size=0.15)

            lgbm_classifier = LGBMModel(categorical_cols=categorical_features)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scale_pos_weight = (y == 0).sum() / (y == 1).sum()

            base_params = {
                "boosting_type": "gbdt",
                "objective": "binary",
                "metric": 'auc',
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "scale_pos_weight": scale_pos_weight
                }

            
            oof_preds, models, best_iterations = lgbm_classifier.train(
                cv=cv, X=X, 
                y=y, params=base_params
                )

            return best_iterations, base_params, oof_preds, models, X_train, X_val, X_test, y_train, y_val, y_test
        
        except Exception as e:
            print("Error occured during the building pipeline of LGBM as :", str(e))
            raise e

    
    def estimator(self) -> Tuple[float, float, 
                                dict, np.ndarray
                                ]:

        """
        The final estimator for the prediction with
        the parameters that fit well taken from the 
        cross validation fitting
        
        Returns:
            tuple(float, float, dict, np.ndarray): Average precision score, roc auc score, 
                                                   classification report, confusion matrix

        """
        
        try:

            best_iterations, params, _, _, X_train, X_val, X_test, y_train, y_val, y_test = self.run_pipeline()
            best_iterator = round(np.median(best_iterations))

            
            X_train = pd.concat([X_train, X_val])
            y_train = pd.concat([y_train, y_val])

            categorical_features = X_train.select_dtypes(exclude=['number']).columns.tolist()

            for cols in categorical_features:
                X_train[cols] = X_train[cols].astype("category")
            
            train_meta = {
                "features": list(X_train.columns),
                    "categorical_mappings": {
                    col: [str(c) for c in X_train[col].cat.categories.tolist()]
                    for col in categorical_features
                    },
                    "dtypes": {col: str(X_train[col].dtype) for col in X_train.columns
                    }
                
                }

            final_model = lgb.LGBMClassifier(
                **params, n_estimators=best_iterator, 
                random_state=42
            )


            final_model.fit(
                X_train, y_train,
                categorical_feature=categorical_features,
                eval_metric='auc'
                )
            
            
            best_f1, best_threshold, average_precision, roc_auc, clf_report, conf_matrx = final_model_prediction(
                final_model, X_test, y_test, best_iterator
            )

            self._save(
                save_dir=self.save_path, final_model=final_model,
                params=params, best_threshold=best_threshold, 
                best_iterator=best_iterator, best_f1=best_f1,
                average_precision=average_precision, 
                roc_auc=roc_auc, train_meta=train_meta
            )

            return average_precision, roc_auc, clf_report, conf_matrx

        except Exception as e:
            print("Error occured during building the final estimator of LGBM as :", str(e))
            raise e


    def _save(
        self, save_dir: str,
        final_model: lgb.LGBMClassifier,
        params: dict, best_threshold: float,
        best_iterator: int, best_f1: float,
        average_precision: float, roc_auc: float,
        train_meta: dict
        ):

        """
        Saving the model with threshold and parameters
        for reusing it in production
        
        Args:
            save_dir (str): The save dir path to which the model is saved
            final_model: The LGBM classifier final model estimator
            params (dict): The parameters used by the final model 
            best_threshold (float): The best threshold used to get the best performance
            best_iterator (int): The best iterator for saving into as metadata of estimator
            best_f1 (float): The final f1 score
            average_precision (float): Final average precision score 
            roc_auc (float): Final roc-auc score
            train_meta (dict): The training metadata

        """

        try:
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            joblib.dump(
                final_model,
                os.path.join(save_dir, "final_estimator.pkl")
                )
            
            with open(os.path.join(save_dir, "final_threshold.json"), 'w') as f:
                json.dump({"Threshhold": float(best_threshold)}, f)

            with open(os.path.join(save_dir, "estimator_metadata.json"), "w") as f:
                json.dump({
                    "best iterator": best_iterator,
                    "parameters": params,
                    "best f1": best_f1,
                    "roc auc score": roc_auc,
                    "average precision": average_precision
                }, f)

            with open(os.path.join(save_dir, "training_meta.json"), "w") as f:
                json.dump(train_meta, f)
               
           

        except Exception as e:
            print("Error occured while saving artifacts as:", str(e))
            raise e


