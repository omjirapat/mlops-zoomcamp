import os
import pickle
import mlflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("nyc-taxi-experiment")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    lr, dv, intercept_, X_train, y_train = data
    current_working_directory = os.getcwd()
    print(current_working_directory)
    with open("/home/src/mlops/homework_03/models/dv.b", "wb") as f_out:
        pickle.dump(dv, f_out)

    with mlflow.start_run():
        # lr.fit(X_train, y_train)
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        mlflow.log_artifact("/home/src/mlops/homework_03/models/dv.b", artifact_path="DictVectorizer")



