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

    def _clean_data(self) -> pd.DataFrame:
        """
        Clean the Data
        
        """

        try:

            numeric_features = self.data.select_dtypes(include=['number']).columns.tolist()
            categorical_features = self.data.select_dtypes(exclude=['number']).columns.tolist()

            print("Cleaning the Data")

            if 'TotalCharges' in self.data.columns:

                self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')

                self.data['TotalCharges'] = self.data['TotalCharges'].fillna(self.data['TotalCharges'].median())


            for cols in categorical_features:
                self.data[cols] = self.data[cols].astype('category')
            
            return self.data

        except Exception as e:
            print("Error in cleaning data:", e)
            raise e

    def feature_engineer(self):

        """
        
        Feature engineer the dataset features for better learning
        space for model
        
        """
        
        try:
            
            data = self._clean_data()

            if 'MonthlyCharges' and 'total_addons' in self.data.columns:
                
                data['total_addons'] = data['total_addons'].fillna(0)
                data['spend_per_addon'] = data['MonthlyCharges'] / (1 + data['total_addons'])

                return data

        except Exception as e:
            print("Error occured during feature engineering as :", str(e))
            raise e
                

        
    