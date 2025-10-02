from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import pandas as pd
import numpy as np
import optuna
import os
import joblib


if not os.path.exists("models/"):
    os.makedirs("model")


class LGBMModel:

    def __init__(
        self, categorical_cols: list[str]
        ):

        """
        Initialize the model class with parameters
        
        Args:
            numeric_cols (list[str]): The list of numeric columns
            categorical_cols (list[str]): The list of categorical columns
            
        """

        self.categorical_cols = categorical_cols

    
    def train(
        self, cv, X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        params: dict
        ):

        """
        Train the lightgbm model with parameters with the train
        splits of X and y
        
        Args:
            cv: StratifiedKFold or int like 3 or 5. StratifiedKFold 
                will be more efficient
            X (pd.DataFrame): The X train for training the lightgbm model
            y (pd.Series | np.ndarray): The target y train for training the 
                                        lightgbm model
            params (dict): The parameters for the lightgbm model
        
        Returns:
            oof_preds, oof_per_folds models: The out-of-folds predictions, 
            the per fold predictions and models

        """

        try:

            oof_preds = np.zeros(len(X))
            oof_per_preds = []

            models = []
            fold_metrics = []
            best_iterations = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):

                print(f"Fold {fold+1}")

                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                lgb_classifier = lgb.LGBMClassifier(
                    **params, n_estimators=1000, random_state=42
                )

                lgb_classifier.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    categorical_feature=self.categorical_cols,
                    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(0)]
                )

                val_probs = lgb_classifier.predict_proba(
                    X_val, num_iteration=lgb_classifier.best_iteration_)[:, 1]

                oof_preds[val_idx] = val_probs
                oof_per_preds.append(val_probs)
                models.append(lgb_classifier)
                best_iterations.append(lgb_classifier.best_iteration_)


                val_auc = roc_auc_score(y_val, val_probs)
                val_ap = average_precision_score(y_val, val_probs)

                print(f"Fold: {fold+1} | AUC Score: {val_auc:.4f} | AP: {val_ap:.4f}")
                fold_metrics.append(
                    {"fold": fold,
                    "AUC": val_auc,
                    "AP": val_ap
                    })

            return oof_preds, models, best_iterations

        except Exception as e:
            print("Error occured in LGBM model fitting as:", str(e))
            raise e
    
    
    def _objective(
        self, cv, trial: int,
        scoring: str, params: dict,
        X: pd.DataFrame, 
        y: pd.Series | np.ndarray
        ):

        """
        Objective for optuna parameter tuning
        
        Args:
            cv: StratifiedKFold or int 
            scoring (str): The scoring method inside cross val
            params (dict): The parameters for hyperparameter tuning
        
        Returns:
            score: The cross val score

        """
        
        try:

            classifier = lgb.LGBMClassifier(**params, n_estimators=200, random_state=42)
            score = cross_val_score(classifier, X, y, scoring=scoring, cv=cv, n_jobs=-1).mean()
        
        except Exception as e:
            print("Error occured during cv scoring for lightgbm as:", str(e))
            raise e

        return score


    def train_optuna(
        self, cv,
        X: pd.DataFrame, y:pd.DataFrame,
        n_trails: int, ):

        """
        Train the lightgbm model with optuna hyperparameter
        strategy

        Args:
            cv: The folds to train on, can be int or StratifiedKFold
            X (pd.DataFrame): The X set to train inside folds
            y (pd.DataFrame): The y set to train inside folds
            n_trials (int): The number of trails the optuna to run
        
        Returns:
            oof_preds, oof_per_folds models: The out-of-folds predictions, 
            the per fold predictions and models
        
        """

        try:

            study = optuna.create_study(direction='maximize')
            study.optimize(
                self._objective, n_trials=n_trails,
                n_jobs=-1
            )

            best_parameter = study.best_params

            oof_preds = np.zeros(len(X))
            oof_per_preds = []

            models = []
            fold_metrics = []
            best_iterations = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):

                print(f"Fold {fold+1}")

                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                lgb_classifier = lgb.LGBMClassifier(
                    **best_parameter, n_estimators=1000, random_state=42
                )

                lgb_classifier.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    categorical_feature=self.categorical_cols,
                    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(0)]
                )

                val_probs = lgb_classifier.predict_proba(
                    X_val, num_iteration=lgb_classifier.best_iteration_)[:, 1]

                oof_preds[val_idx] = val_probs
                oof_per_preds.append(val_probs)
                models.append(lgb_classifier)
                best_iterations.append(lgb_classifier.best_iteration_)

                val_auc = roc_auc_score(y_val, val_probs)
                val_ap = average_precision_score(y_val, val_probs)

                print(f"Fold: {fold+1} | AUC Score: {val_auc:.4f} | AP: {val_ap:.4f}")
                fold_metrics.append(
                    {"fold": fold,
                    "AUC": val_auc,
                    "AP": val_ap
                    })

            return oof_preds, models, best_iterations
        
        except Exception as e:
            print("Error occured during train optuna as:", str(e))
            raise e


    


                


