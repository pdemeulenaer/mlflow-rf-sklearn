# #from flask import Flask, request
# from flask import Flask, request, redirect, url_for, flash, jsonify
# import pickle
# #from numpy import array2string
# import numpy as np

# http://mlflow-rf-sklearn-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/scorebis?petal_length=4.0&petal_width=1.0&sepal_length=1.0&sepal_width=1.0


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

# COMMAND ----------

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

data_conf, model_conf

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
#mlflow.set_tracking_uri('http://mlflow-server.local')
mlflow.set_tracking_uri('http://mlflow-service-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/')
mlflow.set_experiment("rf-sklearn_experiment")
# mlflow.set_experiment("/Shared/simple-rf-sklearn/simple-rf-sklearn_experiment")

# Setting the requried environment variables
#os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://mlflow-minio.local' #'http://mlflow-minio-service-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/minio/' #'http://mlflow-minio.local/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://mlflow-minio-service-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/' #'http://mlflow-minio.local/'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin' #'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin' #'minio123'


# ---------------------------------------------------------------------------------------
# Main TRAINING Entry Point
# ---------------------------------------------------------------------------------------
def train(data_conf, model_conf, **kwargs):

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


# define the app
app = Flask(__name__)


# TRAINING
@app.route('/train', methods=['GET'])
def dfjson():
    """
    return a json representation of the dataframe
    """
    string = train(data_conf, model_conf) 
    return string #Response(df.to_json(orient="records"), mimetype='application/json')  


# SCORING
# use decorator to define the /score input method and define the predict function
@app.route("/score", methods=["POST", "GET"])
def predict_species():
    # create list and append inputs
    flower = []
    flower.append(request.args.get("petal_length"))
    flower.append(request.args.get("petal_width"))
    flower.append(request.args.get("sepal_length"))
    flower.append(request.args.get("sepal_width"))

    # Define the path to the pickled model
    model_path = "rf-model.pkl"

    # Unpickle the random forest model
    with open(model_path, "rb") as file:
        unpickled_rf = pickle.load(file)  

    # Return the prediction
    return np.array2string(unpickled_rf.predict([flower]))

# use decorator to define the /score input method and define the predict function
@app.route("/scorebis", methods=["POST", "GET"])
def predict_species_bis():
    # Create list and append inputs
    flower = []
    flower.append(request.args.get("petal_length"))
    flower.append(request.args.get("petal_width"))
    flower.append(request.args.get("sepal_length"))
    flower.append(request.args.get("sepal_width"))

    # Define the path to the pickled model
    model_path = "rf-model.pkl"

    # Unpickle the random forest model
    with open(model_path, "rb") as file:
        unpickled_rf = pickle.load(file)  

    # Return the prediction
    return np.array2string(unpickled_rf.predict([flower]))    

@app.route('/api/', methods=['POST'])
def makecalc():
    # Get the data
    data = request.get_json()

    # Define the path to the pickled model
    model_path = "rf-model.pkl"

    # Unpickle the random forest model
    with open(model_path, "rb") as file:
        unpickled_rf = pickle.load(file)   
    
    # Make the predictions 
    prediction = np.array2string(unpickled_rf.predict(data))
    # return the prediction    
    return jsonify(prediction)    


# run the app
if __name__ == "__main__":

    app.run(host="0.0.0.0", debug=True, port="8080")
