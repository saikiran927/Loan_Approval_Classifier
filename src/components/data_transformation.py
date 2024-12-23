import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.components.data_ingestion import DataIngestion
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

mdl=DataIngestion('Dataset/loan_data.csv')
train_path,test_path=mdl.splitting()
class DataTransformation:
    def __init__(self,train_path,test_path):
        self.train_path=train_path
        self.test_path=test_path
    def data_transformer_obj(self):
        num_cols=df.select_dtypes(exclude='object')
        cat_cols=df.select_dtypes(include='object')
        
        
        num_pipeline=Pipeline(steps=[('Imputer',SimpleImputer(strategy='median')),
               ('Standard_Scaler',StandardScaler())])
        cat_pipeline=Pipeline(steps=[("Imputer",SimpleImputer(strategy='most_frequent')),
                                     ('Encoder',OneHotEncoder()),
                                     ('Scaler'),StandardScaler()])
        preprocessor=ColumnTransformer([("numerical",num_pipeline,num_cols),
                                        ("categorical",cat_pipeline,cat_cols)])
        return preprocessor
        
    
     