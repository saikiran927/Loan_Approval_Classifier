import os
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier,XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,roc_auc_score
from src.components.data_transformation import DataTransformation


@dataclass
class Trained_Model_path:
    model_path=os.path.join("artifacts","model.pkl")
    

class ModelTrainer:
    def __init__(self):
        self.model_paths=Trained_Model_path()
    def Initiate_model_training(self,trainarr,testarr):
        models={"LogisticRegression":LogisticRegression,
                "KNeighborsClassifier":KNeighborsClassifier,
                "Naive Bayes classifier":GaussianNB,
                "SVC":SVC,
                "DecisionTreeClassifier":DecisionTreeClassifier,
                "RandomForestClassifier":RandomForestClassifier,
                "GradientBoostingClassifier":GradientBoostingClassifier,
                "AdaBoostClassifier":AdaBoostClassifier,
                "XGBClassifier":XGBClassifier,
                "XGBRFClassifier":XGBRFClassifier,
                "LGBMClassifier":LGBMClassifier,
                "CatBoostClassifier":CatBoostClassifier
                }
        results=evaluate_model(trainarr[:,:-1],trainarr[:,-1],testarr[:,:-1],testarr[:,-1],models=models)
        
def evaluate_model(xtrain,ytrain,xtest,ytest,models):
    for i in range(len(list(models))):
        model=list(models.values())[i]
        model.fit(xtrain,ytrain)
        ypred_test=model.predict(xtest)
        
        
        
