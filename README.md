# rf-sklearn

## Intro

This is a simple random forest classifier of the Iris dataset that is trained (so far locally), then the pickle is moved (so far manually) to a "production" folder, and therein a flask app is created to serve the model using the pickle file.

The flask app docker image is deployed as a container on Openshift (tested either Kubernete's local Minikube or Redhat's Openshift Sandbox, in the latter one so far the deployment is done in the UI using the Dockerfile deployment option).

When using the Dockerfile deployment option, you need to select a route creation, so that you can then serve the model using the address, such as this example:

http://mlflow-rf-sklearn-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/score?petal_length=4.0&petal_width=1.0&sepal_length=1.0&sepal_width=1.0

This so far, when deployed, returns in the browser: [1.]

## To-Do list

* Create an Azure DevOps CI pipeline. The CI will contain traditional code analysis (Pylint), Unit testing (pytest, pytest-cov) and bring reports to SonarCloud (where only Master branch is considered...for a real Git Flow, use SonarQube) [Todo] 

* Create an Azure DevOps CD pipeline. The CD needs to deploy the flask app to the Openshift Sandbox. I need to figure out how to make the connection seamless [Todo]

* Create multiple functionalities for the Flask app. For example: training, serving, ... see https://medium.com/geekculture/machine-learning-prediction-in-real-time-using-docker-python-rest-apis-with-flask-and-kubernetes-fae08cd42e67 as example

* Train the model as a scheduled job [Todo]

* Customize the app to ingest arrays [Todo]

* Create a streamlit application to monitor the app [Todo] (or use Grafana)

* Later, create multiple environments and train/test in those, before migrating to PROD [Todo]

# Later development ideas

* Later, create another app running in Openshift that will create data live. For this I would think of a make_blobs function of scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html). The app would generate such datapoints for let's say 3 clusters every 10s. Then these datapoints would be loaded to some postgres database. Then both training and serving codes would be able to query that db

* Also, we could test an app that would serve in time windows, with windows ranging from taking the last batch of data and serving it, to taking the last day or month of data and serving it... would be interesting to see the behaviour

* Instead of trying to pass in the CD step directly the code to Openshift, build the docker image from the dockerfile and push it to a docker registry (like DockerHub). Then Openshift should "feel" the change in the Docker registry and re-deploy the app. The deployment would then use a yaml manifest(?)

* Create the MDLC feedback loop of MLOps: when the code changes, or the data change "significantly" (to be defined), or the model performance (accuracy here) drops under a threshold (to be defined), would re-trigger the training of the model and deploy the dockerfile.


