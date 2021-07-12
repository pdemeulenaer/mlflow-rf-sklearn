#from flask import Flask, request
from flask import Flask, request, redirect, url_for, flash, jsonify
import pickle
#from numpy import array2string
import numpy as np

# http://mlflow-rf-sklearn-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/scorebis?petal_length=4.0&petal_width=1.0&sepal_length=1.0&sepal_width=1.0


# define the app
app = Flask(__name__)


# use decorator to define the /score input method and define the predict function
@app.route("/score", methods=["POST", "GET"])
def predict_species():
    # create list and append inputs
    flower = []
    flower.append(request.args.get("petal_length"))
    flower.append(request.args.get("petal_width"))
    flower.append(request.args.get("sepal_length"))
    flower.append(request.args.get("sepal_width"))
    # return the prediction
    return np.array2string(unpickled_rf.predict([flower]))

# use decorator to define the /score input method and define the predict function
@app.route("/scorebis", methods=["POST", "GET"])
def predict_species_bis():
    # create list and append inputs
    flower = []
    flower.append(request.args.get("petal_length"))
    flower.append(request.args.get("petal_width"))
    flower.append(request.args.get("sepal_length"))
    flower.append(request.args.get("sepal_width"))
    # return the prediction
    return np.array2string(unpickled_rf.predict([flower]))    

@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    prediction = np.array2string(unpickled_rf.predict(data))
    # return the prediction    
    return jsonify(prediction)    


# run the app
if __name__ == "__main__":

    # define the path to the pickled model
    model_path = "rf-model.pkl"

    # unpickle the random forest model
    with open(model_path, "rb") as file:
        unpickled_rf = pickle.load(file)

    app.run(host="0.0.0.0", debug=True, port="8080")
