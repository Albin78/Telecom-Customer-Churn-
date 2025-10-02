from preprocessing.preprocess import PreprocessSetup
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    average_precision_score, roc_auc_score,
    precision_recall_curve
    )
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipe
from imblearn.over_sampling import SMOTE
import pandas as pd



class LRModel:

    def __init__(self, data: pd.DataFrame,
                numeric_features: list[str],
                categorical_features: list[str]
                ):

        self.data = data
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
    
    
    def gridsearch_lr(self, cv, params: dict, model_type: str,
                     scoring: str='average_precision',
                      n_jobs: int=-1):
        
        """
        The GridSearch Hyperparameter CV method for fitting the
        model in different folds and validates the performance.
        
        Args:
            params (dict): The parameters for hyperparameter tuning
            scoring (str): The scoring metric. Defaults to average precision
            cv (int): The cross validation folds. Defaults to 5
            n_jobs(int): Parallel processing. Defaults to -1
            
        Returns:
        
        """
        
        try:
            
            preprocess = PreprocessSetup(
                data=self.data, numeric_features=self.numeric_features,
                cat_bin_features=self.categorical_features
                )
            
            
            
            model_pipe = preprocess.model_pipeline(model_type=model_type)
            
            gridsearch_cv = GridSearchCV(
                    model_pipe, params, cv=cv,
                    scoring=scoring, n_jobs=n_jobs
                    )

            return gridsearch_cv
        
        except Exception as e:
            print("Error occured during building grid search CV as :", str(e))
            raise e


    