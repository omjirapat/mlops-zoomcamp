mlflow-ui:
	cd project && nohup mlflow ui --backend-store-uri sqlite:///mlflow.db > mlflow.log 2>&1 &

prefect:
	cd project && prefect server start

all: mlflow-ui prefect