import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
import joblib
import pandas as pd
import pycaret
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 7),
    'email ':['castro.sebastian@pucp.pe'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_workflow_SCASTRO',
    default_args=default_args,
    description='A simple ML pipeline',
    schedule_interval='0 17 * * *',
)

def load_data():
 df = pd.read_csv('train.csv',sep=",")
 X=df.drop(columns=["Target"])
 y=df["Target"]
 X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
 return X_train.tolist(), y_train.tolist(), X_test.tolist(), y_test.tolist()

def preprocess_data(ti):
    X, y = ti.xcom_pull(task_ids='load_data')

    model = joblib.load('/opt/airflow/dags/final_dt_.joblib')
    model.fit(X, y)
    model_path = '/opt/airflow/dags/final_dt_.joblib'
    dump(model, model_path)
    return model_path

def evaluate_model(ti):
    model_path = ti.xcom_pull(task_ids='preprocess_data')
    model = load(model_path)
    X, y = ti.xcom_pull(task_ids='preprocess_data')
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return accuracy
    

GetDataKaggle = PythonOperator(
    task_id='GetDataKaggle',
    python_callable=load_data,
    dag=dag,
)

AutoML_PyCaret = PythonOperator(
    task_id='AutoML_PyCaret',
    python_callable=preprocess_data,
    dag=dag,
)

SubmitKaggle = PythonOperator(
    task_id='SubmitKaggle',
    python_callable=evaluate_model,
    dag=dag,
)

GetDataKaggle >> AutoML_PyCaret >> SubmitKaggle
