from preprocessing.data_split import DataSplit
from preprocessing.ingest_clean import IngestData, CleanData
from models.LGBM import LGBMModel
from sklearn.model_selection import StratifiedKFold
from models.metrics import evaluation_metrics_lgbm


def main():

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
    
    best_threshold, best_f1, avg_precision, roc_auc, clf_report, conf_matrix = evaluation_metrics_lgbm(
        X_test=X_test, y=y, y_test=y_test, 
        models=models, oof_preds=oof_preds
    )

    print("Best Threshold:", best_threshold)
    print("Best F1:", best_f1)
    print("Average Precision:", avg_precision)
    print("AUC Score:", roc_auc)
    print("\nClassification Report:\n\n", clf_report)
    print("\nConfusion Matrix:\n", conf_matrix)



if __name__ == "__main__":    
    main()




    
    

