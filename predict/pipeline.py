from preprocessing.data_split import DataSplit
from preprocessing.ingest_clean import IngestData, CleanData
from models.LGBM import LGBMModel
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import numpy as np
import pandas as pd

class LGBMPipeline:

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

        ingest_data = IngestData(file_path=r"dataset\feature_engineered_data.csv").get_data()
        data = CleanData(data=ingest_data).feature_engineer()

        categorical_features = data.select_dtypes(exclude='number').columns.tolist()

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

        return best_iterations, base_params, oof_preds, models, X_train, X_val, y_train, y_val

    
    def estimator(self):

        """
        The final estimator for the prediction with
        the parameters that fit well taken from the 
        cross validation fitting
        
        """

        best_iterations, params, _, _, X_train, X_val, y_train, y_val = self.run_pipeline()
        best_iterator = int(np.mean(best_iterations))

        
        X_train = pd.concat([X_train, X_val])
        y_train = pd.concat([y_train, y_val])

        categorical_features = X_train.select_dtypes(exclude=['number']).columns.tolist()

        for cols in categorical_features:
            X_train[cols] = X_train[cols].astype("category")


        final_model = lgb.LGBMClassifier(
            **params, n_estimators=best_iterator, 
            random_state=42
        )


        final_model.fit(
            X_train, y_train,
            categorical_feature=categorical_features,
            eval_metric='auc'
            )
        
        



    




    
    

