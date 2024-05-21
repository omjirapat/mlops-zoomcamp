mlflow-ui:
	cd W2 && nohup mlflow ui --backend-store-uri sqlite:///mlflow.db

jupyter-lab:
	cd W2 && jupyter lab

all: mlflow-ui jupyter-lab