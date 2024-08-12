## Santander Bicycle Rentals in London Prediction
This project focuses on predicting bicycle rental demand in London using the Santander dataset. By applying machine learning, the goal is to optimize bike-sharing systems by ensuring efficient bicycle availability. The process includes data preprocessing, feature engineering, model training, and deployment, all integrated through MLOps to create a scalable and robust prediction pipeline.

## Solution

The solution involves the following steps for code execution:

### Set up

1. Create a Python virtual environment and execute `pip install -r requirements.txt` to install the required dependencies.

### Step 1: Start MLflow and Prefect Services

2. Start the MLflow UI with a backend store using SQLite:
    ```sh
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    ```

3. In a separate terminal, start the Prefect server:
    ```sh
    prefect server start
    ```
These commands will initiate the necessary services to monitor and manage machine learning workflows effectively.

### Step 2: Hyperparameter Optimization and Model Registration

4. Run the following command to perform hyperparameter optimization:
    ```sh
    python hpo.py
    ```

5. Run the following command to register the best model into the MLflow service:
    ```sh
    python register_model.py
    ```

6. Run the following command to train and export the model locally:
    ```sh
    python train.py
    ```

### Step 3: Batch Deployment

7. Run the following command to perform batch deployment, ensuring that the corresponding files are present in the `data` folder:
    ```sh
    python batch_predict.py {year} {season}
    ```

This command will generate batch predictions based on the specified year and season, using the prepared data files in the `data` folder. The prediction results will be exported to the `output` folder.

### Step 4: Monitoring

8. Start the Evidently UI:
    ```sh
    evidently ui
    ```

9. Run the following command to perform monitoring:
    ```sh
    python monitoring.py
    ```