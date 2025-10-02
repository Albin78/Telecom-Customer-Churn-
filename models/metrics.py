from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, classification_report, 
    confusion_matrix
)
import pandas as pd
import numpy as np



def validation_metrics(
    self, X_test: pd.DataFrame,
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

        return auc_score, avg_precision, clf_report, conf_matrx, best_threshold, best_f1

    except Exception as e:
        print("Error occured during the evaluation metrics as :", str(e))
        raise e
        