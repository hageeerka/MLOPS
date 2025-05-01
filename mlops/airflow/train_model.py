import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor,LinearRegression, Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
from mlflow.models import infer_signature
import joblib


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train():
    salary = pd.read_csv("./salary.csv")
    
    X = salary[['Experience Years']]
    y = salary['Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2529)
    
    #SGDRegressor
    params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1 ],
              'l1_ratio': [0.001, 0.05, 0.01, 0.2],
              'max_iter': [1000, 2000, 5000]
     }
    
    mlflow.set_experiment('simple models')
    
    with mlflow.start_run():
    
        lr = SGDRegressor(random_state=42)
    
        clf = GridSearchCV(lr, params, cv = 5, n_jobs=4)
        clf.fit(X_train, y_train)
        best = clf.best_estimator_
        y_pred = best.predict(X_test)
        y_price_pred = y_pred
        (rmse, mae, r2)  = eval_metrics(y_test, y_price_pred)
        
        alpha = best.alpha
        l1_ratio = best.l1_ratio
        max_iter = best.max_iter
        
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("max_iter", max_iter)
        
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        with open("lr_salary.pkl", "wb") as file:
            joblib.dump(best, file)
