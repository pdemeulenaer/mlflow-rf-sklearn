from flask import Flask, request, redirect, url_for, flash, jsonify, send_file, make_response, Response
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import requests
import json
import io


# ==============================
# 1. Full Dataset Loading
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



# define the app
app = Flask(__name__)



# @app.route('/api/')
# def makecalc():
#     resp = make_response(data_pd.to_csv())
#     # resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
#     # resp.headers["Content-Type"] = "text/csv"
#     return resp


@app.route("/api")
def dfjson():
    """
    return a json representation of the dataframe
    """
    # df = get_dataframe_from_somewhere()
    return Response(data_pd.to_json(orient="records"), mimetype='application/json')    

# run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port="8080")    