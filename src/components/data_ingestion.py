import os 
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
# from src.exception import CustomException

@dataclass
class DataIngestionConfig():
    train_path=os.path.join('artifacts','train.csv')
    test_path=os.path.join('artifacts','test.csv')
    
class DataIngestion:
    def __init__(self,path):
        self.config=DataIngestionConfig()
        self.data_path=path
        
    def splitting(self):
        df=pd.read_csv(self.data_path)
        train_df,test_df=train_test_split(df,test_size=0.20,random_state=42)
        os.makedirs(os.path.dirname(self.config.train_path),exist_ok=True)
        train_df.to_csv(self.config.train_path,index=False)
        os.makedirs(os.path.dirname(self.config.test_path),exist_ok=True)
        test_df.to_csv(self.config.test_path,index=False)
        return (
            self.config.train_path,
            self.config.test_path
        )
        
    

mdl=DataIngestion('Dataset/loan_data.csv')
mdl.splitting()