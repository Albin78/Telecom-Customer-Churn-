# 🧠 Customer Churn Prediction System


## 📘 Overview

This project is an end-to-end Customer Churn Prediction System designed to predict whether a customer is likely to churn (leave) or stay with the service.
It includes complete stages of a real-world ML workflow — from data ingestion, feature engineering, and model training to evaluation, artifact saving, and deployment via Streamlit.

The primary objective is to help businesses identify potential churners and take proactive measures for retention.

⚙️ Project Structure


Customer Churn/
│
├── app/
│   ├── predict_app.py       
│   └── run_app.py            
├── dataset/
|   ├── feature_engineered_data.csv        
│   └── raw_dataset.csv
|  
├── notebooks/
│   ├── EDA1.ipynb            
│   ├── EDA2.ipynb            
│   ├── EDA3.ipynb            
│   ├── LR_model.ipynb         
│   └── Tree_model.ipynb       
│
├── models/
│   ├── LGBM.py               
│   ├── metrics.py            
│   └── model.py             
│
├── preprocessing/
│   ├── data_split.py         
│   ├── ingest_clean.py       
│   └── preprocess.py         
│
├── predict/
│   ├── pipeline.py           
│   └── predict_model.py      
│
├── artifacts/
│   ├── final_estimator.pkl       
│   ├── estimator_metadata.json   
│   |── final_threshold.json      
│   └── training_meta.json        
│   
├── requirements.txt          
└── README.md                 


### 🚀 Key Features

✅ Complete ML Workflow — Data ingestion → cleaning → feature engineering → model training → evaluation
✅ Cross-validation (CV) + Optuna — For robust and optimized LightGBM model performance
✅ Artifact Management — Model and metadata saving for reproducibility
✅ Metrics Module — Includes accuracy, precision, recall, F1, AUC, and threshold tuning
✅ Interactive Streamlit UI — For real-time churn predictions
✅ Modular Design — Each component (data, model, prediction, UI) is isolated for scalability

### 🧩 Model Used

The project implements multiple models (Logistic Regression, Random Forest, LightGBM).
After extensive evaluation, LightGBM was chosen as the final model because it consistently outperformed others in metrics like:

Higher AUC and Average Precision

Faster training time

Better handling of feature interactions

## 🧰 Installation & Setup

## Clone the repository
`git clone https://github.com/Albin78/Telecom-Customer-Churn-.git`

`cd customer-churn-prediction`

## Create & activate a virtual environment
`python -m venv .venv`

## Activate environment

### On Windows:
`.venv\Scripts\activate`

### On macOS/Linux:
`source .venv/bin/activate`

## Install dependencies
`pip install -r requirements.txt`

## 🧪 Running the Pipeline

▶️ Run the full training pipeline

This command runs ingestion, cleaning, feature engineering, cross-validation, and model fitting.

`python -m predict.pipeline`


Artifacts such as the final model and metadata will be saved in the `models_artifacts/` directory.

🎯 Making Predictions
Using Python


🖥️ Launch the Streamlit App
Run Command
`cd app`
`streamlit run predict_app.py`


The app provides an interactive interface for entering customer details and viewing predictions visually.

## 🎨 Streamlit UI Highlights

Dark blue background theme with a soft overlay

Background image for visual appeal

Styled input components and buttons


Confidence percentage display

Error-handling for model loading and prediction issues

## 📊 Evaluation & Metrics

Performance metrics are implemented in models/metrics.py including:

Accuracy

Precision / Recall / F1

ROC-AUC

Confusion Matrix

Threshold tuning logic (for classification decision boundary optimization)

The final LightGBM model achieved strong performance, outperforming both Logistic Regression and Random Forest models.

## 🔮 Future Enhancements

🧾 More Data Collection: Expanding the dataset for better generalization

🧠 Feature Enrichment: Incorporate behavioral or transactional features

🔍 Explainability: Add SHAP-based model interpretability

☁️ Deployment: Host the model on cloud platforms (AWS, GCP, or Azure)

📈 MLOps Integration: Automate the workflow using ZenML or MLflow

💬 RAG or LLM Integration: For customer retention insights or intelligent assistant modules


## ⚠️ Limitations

The current model is limited by the available dataset size and diversity.

May show bias towards certain categories if data imbalance exists.

The threshold tuning may need re-evaluation with new data.

The Streamlit app currently supports only single-record predictions (batch predictions can be added later).

# 👨‍💻 Author

Albin Shabu
📧 [albinshabu960@gmail.com]
🔗 LinkedIn: [https://www.linkedin.com/in/albin-shabu-37b2a7250/]

## Data-driven insights for smarter customer retention.