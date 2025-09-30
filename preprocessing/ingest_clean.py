import logging
import pandas as pd

class IngestData:
    """Ingest data from a source file"""

    def __init__(self, file_path):
        """Initialize the IngestData class"""

        self.file_path = file_path

    def get_data(self) -> pd.DataFrame:
        """Get the data from the file path"""

        print("Ingesting data from the file path")
        return pd.read_csv(self.file_path)



class CleanData:
    """Clean the Data"""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the CleanData class
        
        Args:
            data (pd.DataFrame): the dataset for cleaning
            """

        self.data = data

    def clean_data(self) -> pd.DataFrame:
        """
        Clean the Data
        
        """

        try:
            
            binary_features = ["Partner", "Dependents", "PaperlessBilling", "SeniorCitizen"]
            numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            
            categorical_features = [
                "InternetService", "Contract", "PaymentMethod",
                "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
            ]

            cat_bin = categorical_features + binary_features

            print("Cleaning the Data")

            self.data['PaymentMethod'] = self.data['PaymentMethod'].replace({
                    "Bank transfer (automatic)": "Automatic",
                    "Credit card (automatic)": "Automatic",
                    "Electronic check": "Electronic check",
                    "Mailed check": "Mailed check"
            })

            self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')

            self.data['TotalCharges'] = self.data['TotalCharges'].fillna(self.data['TotalCharges'].median())

            self.data[numeric_features] = self.data[numeric_features].astype(float)

            for col in self.data.select_dtypes('object'):
                self.data[col] = self.data[col].str.strip()

            for cols in cat_bin:
                self.data[cols] = self.data[cols].astype('category')

        except Exception as e:
            print("Error in cleaning data:", e)
            raise e
            