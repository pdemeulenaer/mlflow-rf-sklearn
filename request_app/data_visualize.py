from flask import Flask, request, redirect, url_for, flash, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
#import seaborn as sns
import requests
import json
import io

def do_plot():
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

    # Feature selection
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target       = 'label'   

    X = data_pd[feature_cols].values
    y = data_pd[target].values

    #print(X)

    # =======================================
    # 2. Data Generation (with random noise)
    # =======================================
    # N=30

    # data_gen_pd = data_pd[:N]
    # X_gen = data_gen_pd[feature_cols].values
    # y_gen = data_gen_pd[target].values

    # # Adding gaussian noise
    # X_gen = X_gen + np.random.normal(0,1,(N,4))
    # # print(X_gen)

    url = 'http://192.168.98.175:8080/api'
    #url = 'http://mlflow-rf-sklearn-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/api/'

    r = requests.get(url).json()
    data_gen_pd = pd.DataFrame.from_dict(r)

    X_gen = data_gen_pd[feature_cols].values
    y_gen = data_gen_pd[target].values  


    # =======================================
    # 3. Call Prediction API
    # =======================================

    # #url = 'http://0.0.0.0:8080/api/'
    url = 'http://mlflow-rf-sklearn-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/api/'

    data = X_gen.tolist() #[[5.7, 2.8, 4.1, 1.3],[5.8, 2.6, 4., 1.2],[5.8, 2.6, 4., 1.2]]
    j_data = json.dumps(data)
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=j_data, headers=headers)
    pred=np.array(r.text)
    print('prediction is:', pred)
    print('real is:', y_gen)

    # =======================================
    # 4. Create Visualisation
    # =======================================


    # Create bee swarm plot with Seaborn's default settings
    #sns.swarmplot(x='species',y='petal_length',data=data_pd)
    #fig = plt.figure(1,(5,5))

    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,6))

    #axes[0].plot(data_pd['petal_length'],data_pd['sepal_length'],'o',color='grey')
    axes[0].scatter(data_pd['petal_length'],data_pd['sepal_length'],marker='o',c='grey',s=30,alpha=0.5)    
    axes[0].scatter(data_gen_pd.loc[data_gen_pd['label']==0,'petal_length'].values,data_gen_pd.loc[data_gen_pd['label']==0,'sepal_length'].values,marker='o',c='red',s=50,alpha=0.5,label='setosa')
    axes[0].scatter(data_gen_pd.loc[data_gen_pd['label']==1,'petal_length'],data_gen_pd.loc[data_gen_pd['label']==1,'sepal_length'],marker='o',c='blue',s=50,alpha=0.5,label='versicolor')
    axes[0].scatter(data_gen_pd.loc[data_gen_pd['label']==2,'petal_length'],data_gen_pd.loc[data_gen_pd['label']==2,'sepal_length'],marker='o',c='green',s=50,alpha=0.5,label='virginica')
    axes[0].set_xlabel('petal_length')
    axes[0].set_ylabel('sepal_length')

    #axes[1].plot(data_pd['petal_length'],data_pd['sepal_width'],'o',color='grey')
    axes[1].scatter(data_pd['petal_length'],data_pd['sepal_width'],marker='o',c='grey',s=30,alpha=0.5)  
    axes[1].scatter(data_gen_pd.loc[data_gen_pd['label']==0,'petal_length'],data_gen_pd.loc[data_gen_pd['label']==0,'sepal_width'],marker='o',c='red',s=50,alpha=0.5,label='setosa')
    axes[1].scatter(data_gen_pd.loc[data_gen_pd['label']==1,'petal_length'],data_gen_pd.loc[data_gen_pd['label']==1,'sepal_width'],marker='o',c='blue',s=50,alpha=0.5,label='versicolor')
    axes[1].scatter(data_gen_pd.loc[data_gen_pd['label']==2,'petal_length'],data_gen_pd.loc[data_gen_pd['label']==2,'sepal_width'],marker='o',c='green',s=50,alpha=0.5,label='virginica')    
    axes[1].set_xlabel('petal_length')
    axes[1].set_ylabel('sepal_width')
    plt.legend()
    # plt.show()

    # here is the trick save your figure into a bytes object and you can afterwards expose it via flas
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image


# =======================================
# 5. Export Visualisation to Flask
# =======================================

# define the app
app = Flask(__name__)

@app.route('/plots/iris_data/plot', methods=['GET'])
def plot_in_flask():
    bytes_obj = do_plot()
    
    return send_file(bytes_obj,
                     download_name='plot.png',
                     mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)    

