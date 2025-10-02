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
    
    
    def gridsearch(self, cv, params: dict, model_type: str,
                     scoring: str='average_precision',
                      n_jobs: int=-1, smote: bool=False
                      ):
        
        """
        The GridSearch Hyperparameter CV method for fitting the
        model in different folds and validates the performance.
        
        Args:
            params (dict): The parameters for hyperparameter tuning
            model_type (str): Type of model
            scoring (str): The scoring metric. Defaults to average precision
            cv (int): The cross validation folds. Defaults to 5
            n_jobs(int): Parallel processing. Defaults to -1
            smote (bool): smote pipeline to use. Defaults to False 
            
        Returns:
            GridSearchCV model
        
        """
        
        try:
            
            preprocess = PreprocessSetup(
                data=self.data, numeric_features=self.numeric_features,
                cat_bin_features=self.categorical_features
                )
            
            model_pipe = preprocess.model_pipeline(model_type=model_type,
                                                       smote=smote)
            
            gridsearch_cv = GridSearchCV(
                        model_pipe, params, cv=cv,
                        scoring=scoring, n_jobs=n_jobs,
                        refit=True
                        )

            return gridsearch_cv
        
        except Exception as e:
            print("Error occured during building grid search CV as :", str(e))
            raise e

    
    def randomizedsearch(
        self, cv, n_iters: int,
        params: dict, scoring: str, 
        n_jobs: int, smote: bool, 
        model_type: str, random_state: int=42, 
        ):
        
        """
        Perform the randomized search with the input parameters
        
        Args:
            cv: Integer or can use StratifiedKFold for better internal splits
            n_iters (int): The number of iterations to perform
            params (dict): The parameters for random search hyperparameter tuning
            scoring (str): The scoring method 
            model_type (str): Type of model
            smote (bool): The SMOTE method if want
            n_jobs (int): The parallel processing.
            random_state (int): Random state for reproducability. defaults to 42
        
        Returns:
            RandomizedSearch Cv Model

        """
        
        try:
            preprocess = PreprocessSetup(
                    data=self.data, numeric_features=self.numeric_features,
                    cat_bin_features=self.categorical_features
                    )
            
            model_pipe = preprocess.model_pipeline(
                model_type=model_type, smote=smote
                )

            randomized_search = RandomizedSearchCV(
                model_pipe, params, n_iter=n_iters,
                cv=cv, scoring=scoring, random_state=random_state,
                n_jobs=n_jobs, refit=True
            )

            return randomized_search

        except Exception as e:
            print("Error occured during randomized search cv as :", str(e))
        


        



    