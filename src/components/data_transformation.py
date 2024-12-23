import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# from src.components.data_ingestion import DataIngestion
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import dill



@dataclass
class Data_preprocessor_path:
    processor_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.preprocessorpath=Data_preprocessor_path()
    def data_transformer_obj(self):
        num_cols=['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']

        cat_cols=['person_gender','person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file']
        
        
        num_pipeline=Pipeline(steps=[('Imputer',SimpleImputer(strategy='median')),
               ('Standard_Scaler',StandardScaler())])
        cat_pipeline=Pipeline(steps=[("Imputer",SimpleImputer(strategy='most_frequent')),
                                     ('Encoder',OneHotEncoder()),
                                     ('Scaler',StandardScaler(with_mean=False))])
        preprocessor=ColumnTransformer([("numerical",num_pipeline,num_cols),
                                        ("categorical",cat_pipeline,cat_cols)])
        return preprocessor
    def Initiate_data_transformation(self):
        train_path='artifacts/train.csv'
        test_path='artifacts/test.csv'
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)
        preprocessor_obj=self.data_transformer_obj()
        target_column="loan_status"
        input_feature_train_df=train_df.drop(target_column,axis=1)
        output_feature_train_df=train_df[target_column]
        
        input_feature_test_df=test_df.drop(target_column,axis=1)
        output_feature_test_df=test_df[target_column]
        
        input_feature_train_preprocessor=preprocessor_obj.fit_transform(input_feature_train_df)
        input_feature_test_preprocessor=preprocessor_obj.transform(input_feature_test_df)
        
        trainarr=np.c_[input_feature_train_preprocessor,np.array(output_feature_train_df)]
        testarr=np.c_[input_feature_test_preprocessor,np.array(output_feature_test_df)]
        print("transformation pipeline has completed")
        save_object(self.preprocessorpath.processor_path,preprocessor_obj)
        print("saving the preprocessor object also completed")
        
        return trainarr,testarr
    

def save_object(file_path,obj):
    file_dir=os.path.dirname(file_path)
    os.makedirs(file_dir,exist_ok=True)
    with open(file_path,'wb') as file:
        dill.dump(obj,file)
        
    
    pass
# mdl=DataIngestion('Dataset/loan_data.csv')
# train_path,test_path=mdl.splitting()
transformm=DataTransformation()
transformm.Initiate_data_transformation()

        
        
            
    
     