from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, classification_report, 
    confusion_matrix
)
import pandas as pd
import numpy as np



def validation_metrics(
    X_test: pd.DataFrame,
    y_test: pd.Series, best_estimator
    ):

    """
    Finds the model metrics: Average precision, auc score,
    classification report and confusion matrix with threshold
    tuning which helps to evaluate model performance on fitted model
    
    Args:
        X_test (pd.DataFrame): The validation X set 
        y_test (pd.Series): The validation target set 
        best_estimator: The best estimator after the hyperparameter tuning technique of model
        
    Returns:
        tuple(float, float, dict, array): auc_score, avg_precision, classification report, confusion matrix
        
    """
    try:

        val_probs = best_estimator.predict_proba(X_test)[:, 1]

        prec, rec, threshold = precision_recall_curve(y_test, val_probs)

        f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-6)
        best_idx = np.nanargmax(f1_scores)
        best_threshold = threshold[best_idx]
        best_f1 = f1_scores[best_idx]

        y_preds = (val_probs >= best_threshold).astype(int)

        auc_score = roc_auc_score(y_test, val_probs)
        avg_precision = average_precision_score(y_test, val_probs)

        clf_report = classification_report(y_test, y_preds, output_dict=True)
        conf_matrx = confusion_matrix(y_test, y_preds)

        return best_threshold, best_f1, auc_score, avg_precision, clf_report, conf_matrx

    except Exception as e:
        print("Error occured during the evaluation metrics as :", str(e))
        raise e


def evaluation_metrics_lgbm(
    X_test: pd.DataFrame, 
    y: pd.DataFrame | pd.Series,
    y_test: pd.Series | np.ndarray,
    models: list, oof_preds: list
    ):

    """
    Evaluation metrcis for LGBM model
    
    Args:
        X_test (pd.DataFrame): The test set of X or label
        y_test (pd.Series | np.ndarray): The test set of y or target
        models (list): The list of models from cv fits
        oof_preds (list): The out of fold predictions from cv fits
    
    Returns:
        tuple(float, float, dict, array): auc score, average precision score,
                                          classification report, confusion meatrix
    
    """
    
    try:

        precision, recall, threshold = precision_recall_curve(y, oof_preds)
        f1_score = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-6)

        best_idx = np.nanargmax(f1_score)
        best_threshold = threshold[best_idx]
        best_f1 = f1_score[best_idx]

        preds = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
        y_preds = (preds >= best_threshold).astype(int)

        avg_precision = average_precision_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, preds)

        clf_report = classification_report(y_test, y_preds)
        conf_matrix = confusion_matrix(y_test, y_preds)

        return best_threshold, best_f1, avg_precision, roc_auc, clf_report, conf_matrix

    except Exception as e:
        print("Error durinng the evaluation of LGBM as :", str(e))
        raise e

