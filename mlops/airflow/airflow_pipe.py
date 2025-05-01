import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from pathlib import Path
import os
from datetime import timedelta
from train_model import train

def download_data():
    url = 'https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv'
    salary = pd.read_csv(url)
    salary.to_csv("salary.csv", index=False)
    
    return True

dag_salary = DAG(
    dag_id="train_pipe",
    start_date=datetime(2025, 2, 0),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
#    schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)
download_task = PythonOperator(python_callable=download_data, task_id = "download_data", dag = dag_salary)
train_task = PythonOperator(python_callable=train, task_id = "train_model_salary", dag = dag_salary)
download_task >> clear_task >> train_task
