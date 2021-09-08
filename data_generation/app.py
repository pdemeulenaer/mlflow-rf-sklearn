import io
import os
import sys
import json
import socket
import traceback
import pickle
from flask import Flask, request, redirect, url_for, flash, jsonify, send_file, make_response, Response
import numpy as np
import pandas as pd
import requests


#Import of SKLEARN packages
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

import mlflow
from mlflow.tracking.client import MlflowClient




# ==============================
# 1. Full Dataset Loading
# ==============================
# Loading of dataset

def get_dataframe_from_somewhere(N):
    iris = load_iris()                  #The Iris dataset is available through the scikit-learn API
    idx = list(range(len(iris.target)))
    np.random.shuffle(idx)              #We shuffle it (important if we want to split in train and test sets)
    X = iris.data[idx]
    y = iris.target[idx]
    # Load data in Pandas dataFrame
    data_pd = pd.DataFrame(data=np.column_stack((X,y)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
    data_pd.loc[data_pd['label']==0,'species'] = 'setosa'
    data_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
    data_pd.loc[data_pd['label']==2,'species'] = 'virginica'
    data_pd.head()

    # N=30
    data_gen_pd = data_pd[:N]

    # Feature selection
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target       = 'label'      

    # Adding gaussian noise
    data_gen_pd[feature_cols] = data_gen_pd[feature_cols] + np.random.normal(0,0.2,(N,4))

    return data_gen_pd[:N]


# ---------------------------------------------------------------------------------------
# Main TRAINING Entry Point
# ---------------------------------------------------------------------------------------
def train():  # def train(data_conf, model_conf, **kwargs):

    data_json = '''{
        "TEST": {
            "input_train": "default.iris",
            "input_test": "default.iris_test",
            "output_test": "default.iris_test_scored",
            "input_to_score": "default.iris_to_score",
            "output_to_score": "default.scored"  
        },
        "SYST": {
            "input_train": "test.iris",
            "input_test": "test.iris_test",
            "output_test": "test.iris_test_scored",
            "input_to_score": "test.iris_to_score",
            "output_to_score": "test.scored"        
        },
        "PROD": {
            "input_train": "test.iris",
            "input_test": "test.iris_test",
            "output_test": "test.iris_test_scored",
            "input_to_score": "test.iris_to_score",
            "output_to_score": "test.scored"       
        }
    }'''

    config_json = '''{
        "hyperparameters": {
            "max_depth": "20",
            "n_estimators": "100",
            "max_features": "auto",
            "criterion": "gini",
            "class_weight": "balanced",
            "bootstrap": "True",
            "random_state": "21"        
        }
    }'''

    data_conf = json.loads(data_json)
    model_conf = json.loads(config_json)

    # data_conf, model_conf

    # COMMAND ----------

    # Define the environment (dev, test or prod)
    # env = dbutils.widgets.getArgument("environment")

    # print()
    # print('Running in ', env)    

    # data_conf = json.loads(data_json)
    model_conf = json.loads(config_json)

    # print(data_conf[env])
    print(model_conf)      

    # Define the MLFlow experiment location
    # mlflow.set_tracking_uri('http://localhost:5000/#/') # localhost for debugging inside a docker image
    # mlflow.set_tracking_uri('http://mlflow-server.local/') # Minikube
    mlflow.set_tracking_uri('http://10.100.226.228:5000/') # Minikube    
    # mlflow.set_tracking_uri('http://mlflow-service-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/') # Openshift
    # mlflow.set_tracking_uri('http://20.76.192.95:5000/') # AKS
    mlflow.set_experiment("rf-sklearn_experiment")


    # Setting the required environment variables
    # os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://mlflow-minio.local/' # Minikube
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://10.105.184.162:9000/' # Minikube    
    # os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://mlflow-minio-service-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/' # Openshift
    # os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://20.76.152.86:9000/'  # AKS
    os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin' #'minio'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin' #'minio123'    

    try:
        print()
        print("-----------------------------------")
        print("         Model Training            ")
        print("-----------------------------------")
        print()

        # ==============================
        # 1.0 Data Loading
        # ==============================

        # Loading of dataset
        iris = load_iris()                  #The Iris dataset is available through the scikit-learn API
        idx = list(range(len(iris.target)))
        np.random.shuffle(idx)              #We shuffle it (important if we want to split in train and test sets)
        X = iris.data[idx]
        y = iris.target[idx]

        # Load data in Pandas dataFrame
        data_pd = pd.DataFrame(data=np.column_stack((X,y)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
        data_pd.loc[data_pd['label']==0,'species'] = 'setosa'
        data_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
        data_pd.loc[data_pd['label']==2,'species'] = 'virginica'
        data_pd.head()
        
        # Feature selection
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        target       = 'label'   
        
        X = data_pd[feature_cols].values
        y = data_pd[target].values

        # Creation of train and test datasets
        x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.7, stratify=y) #stratify=y ensures that the same proportion of labels are in both train and test sets! 
        
        # Save test dataset
        test_pd = pd.DataFrame(data=np.column_stack((x_test,y_test)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
        test_pd.loc[data_pd['label']==0,'species'] = 'setosa'
        test_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
        test_pd.loc[data_pd['label']==2,'species'] = 'virginica'
        # test_df = spark.createDataFrame(test_pd)
        # test_df.write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format('test_data_sklearn_rf'))

        print("Step 1.0 completed: Loaded Iris dataset in Pandas")      

    except Exception as e:
        print("Errored on 1.0: data loading")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ========================================
        # 1.1 Model training
        # ========================================
        
        with mlflow.start_run() as run:          

            # Model definition
            max_depth = int(model_conf['hyperparameters']['max_depth'])
            n_estimators = int(model_conf['hyperparameters']['n_estimators'])
            max_features = model_conf['hyperparameters']['max_features']
            criterion = model_conf['hyperparameters']['criterion']
            class_weight = model_conf['hyperparameters']['class_weight']
            bootstrap = bool(model_conf['hyperparameters']['bootstrap'])
            clf = RandomForestClassifier(max_depth=max_depth,
                                    n_estimators=n_estimators,
                                    max_features=max_features,
                                    criterion=criterion,
                                    class_weight=class_weight,
                                    bootstrap=bootstrap,
                                    random_state=21,
                                    n_jobs=-1,
                                    oob_score=True)          
                
            # Fit of the model on the training set
            model = clf.fit(x_train, y_train) 

            # score our model and print the output
            print(x_test.shape)
            predicted = clf.predict(x_test)
            accuracy = accuracy_score(y_test, predicted)
            print(
                "Out-of-bag score estimate: {0:.3f}\n"
                "Mean accuracy score: {1:.3f}".format(clf.oob_score_, accuracy
                )
            )      

            # Saving the model locally
            pickle_save = 'rf-model.pkl'
            with open(pickle_save, 'wb') as file:
                pickle.dump(clf, file)  
            
            # Log the model within the MLflow run
            mlflow.log_param("max_depth", str(max_depth))
            mlflow.log_param("n_estimators", str(n_estimators))  
            mlflow.log_param("max_features", str(max_features))             
            mlflow.log_param("criterion", str(criterion))  
            mlflow.log_param("class_weight", str(class_weight))  
            mlflow.log_param("bootstrap", str(bootstrap))  
            mlflow.log_param("max_features", str(max_features))             
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, 
                                   "model",
                                   registered_model_name="sklearn-rf")                        

        print("Step 1.1 completed: model training and saved to MLFlow")                  

    except Exception as e:
        print("Errored on step 1.1: model training")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e       

    print()   

    return 'nice'  

# # if __name__ == "__main__":
# #     train(data_conf, model_conf) 





# define the app
app = Flask(__name__)

# @app.route('/api/')
# def makecalc():
#     resp = make_response(data_pd.to_csv())
#     # resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
#     # resp.headers["Content-Type"] = "text/csv"
#     return resp

# DATA GENERATION APP
@app.route("/api/data_generation")
def dfjson():
    """
    return a json representation of the dataframe
    """
    df = get_dataframe_from_somewhere(N=30)
    return Response(df.to_json(orient="records"), mimetype='application/json')   


# SCORING APP for single data row
@app.route("/api/score", methods=["POST", "GET"])
def predict_species():
    # create list and append inputs
    flower = []
    flower.append(request.args.get("petal_length"))
    flower.append(request.args.get("petal_width"))
    flower.append(request.args.get("sepal_length"))
    flower.append(request.args.get("sepal_width"))

    print([flower])

    # Define the path to the pickled model
    model_path = "rf-model.pkl"

    # Unpickle the random forest model
    with open(model_path, "rb") as file:
        unpickled_rf = pickle.load(file)     

    # Return the prediction
    return np.array2string(unpickled_rf.predict([flower]))         


# SCORING APP for batch of data in pandas shape
@app.route('/api/score_batch', methods=['POST','GET'])
def makecalc():
    # Get the data
    data = request.get_json()
    print(data)

    # # Define the path to the pickled model
    # model_path = "rf-model.pkl"

    # # Unpickle the random forest model
    # with open(model_path, "rb") as file:
    #     unpickled_rf = pickle.load(file)   
    
    # # Make the predictions 
    # prediction = np.array2string(unpickled_rf.predict(data))

    # Define the MLFlow experiment location
    # mlflow.set_tracking_uri('http://localhost:5000/#/') # localhost for debugging inside a docker image
    # mlflow.set_tracking_uri('http://mlflow-server.local/') # Minikube
    mlflow.set_tracking_uri('http://10.100.226.228:5000/') # Minikube    
    # mlflow.set_tracking_uri('http://mlflow-service-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/') # Openshift
    # mlflow.set_tracking_uri('http://20.76.192.95:5000/') # AKS
    mlflow.set_experiment("rf-sklearn_experiment")

    # Setting the required environment variables
    # os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://mlflow-minio.local/' # Minikube
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://10.105.184.162:9000/' # Minikube    
    # os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://mlflow-minio-service-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/' # Openshift
    # os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://20.76.152.86:9000/'  # AKS
    os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin' #'minio'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin' #'minio123' 

    mlflow_model_name = 'sklearn-rf'
    mlflow_model_stage = 'Staging'
        
    # Detecting the model dictionary among available models in MLflow model registry. 
    client = MlflowClient()
    for mv in client.search_model_versions("name='{0}'".format(mlflow_model_name)):
        if dict(mv)['current_stage'] == mlflow_model_stage:
            model_dict = dict(mv)
            break  
            
    print('Model extracted run_id: ', model_dict['run_id'])
    print('Model extracted version number: ', model_dict['version'])
    print('Model extracted stage: ', model_dict['current_stage'])                
    
    model_path = model_dict['source']      
    print("model_path: ", model_path)      

    # Loading the model from MLflow artifact
    model = mlflow.pyfunc.load_model(model_path)
    # model.predict(model_input)    
    prediction = model.predict(pd.DataFrame(data))
    prediction = np.array2string(prediction)

    # return the prediction    
    return jsonify(prediction)   


# TRAINING APP
@app.route('/api/train', methods=['GET'])
def train_json():
    """
    return a json representation of the dataframe
    """
    string = train() 
    return jsonify(string) #Response(df.to_json(orient="records"), mimetype='application/json')  


# run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port="8080")    