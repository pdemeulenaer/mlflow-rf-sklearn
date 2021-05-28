# rf-sklearn

## Intro

This is a simple random forest classifier of the Iris dataset that is trained (so far locally), then the pickle is moved (so far manually) to a "production" folder, and therein a flask app is created to serve the model using the pickle file.

The flask app docker image is deployed as a container on Openshift (tested either Kubernete's local Minikube or Redhat's Openshift Sandbox, in the latter one so far the deployment is done in the UI using the Dockerfile deployment option).

## To-Do list

* Create an Azure DevOps CI pipeline. The CI will contain traditional code analysis (Pylint), Unit testing (pytest, pytest-cov) and bring reports to SonarCloud (where only Master branch is considered...for a real Git Flow, use SonarQube) [Todo] 

* Create an Azure DevOps CD pipeline. The CD needs to deploy the flask app to the Openshift Sandbox. I need to figure out how to make the connection seamless [Todo]

* Train the model as a scheduled job [Todo]

* Customize the app to ingest arrays [Todo]

* Create a streamlit application to monitor the app [Todo]

* Later, create multiple environments and train/test in those, before migrating to PROD [Todo]

* Later, create another app running in Openshift


