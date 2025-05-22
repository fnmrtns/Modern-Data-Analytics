import logging
import warnings
import mlflow

#set up connection to dagshub for mlflow to mlflow tracking server: 
mlflow.set_tracking_uri("https://dagshub.com/fienme/Modern-Data-Analytics.mlflow/")

# model training with mlflow logging
def train(it_n_estimators, it_learning_rate, it_max_depth):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor
    import json
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    from mlflow.models.signature import infer_signature
    import time
    import joblib
    import os

    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

  
    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    

    warnings.filterwarnings("ignore")
    np.random.seed(40)


    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            f"Unable to download training & test CSV, check your internet connection. Error: {e}"
        )


    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, test_size=0.25, random_state=40)

    train.to_csv("train_data.csv", index=False)
    test.to_csv("test_data.csv", index=False)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Set default values if no n_estimators is provided
    n_estimators = 100 if int(it_n_estimators) is None else int(it_n_estimators)

    # Set default values if no learning_rate is provided
    learning_rate = 0.1 if float(it_learning_rate) is None else float(it_learning_rate)

    # Set default values if no max_depht is provided
    max_depth = 5 if int(it_max_depth) is None else int(it_max_depth)


    # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run(log_system_metrics=True) as run:
        
        # timing is only to record system-metrics:
        time.sleep(15)

        # Execute xgboostregressor with standardscaler as preprocessing in pipeline:

        preprocessor = StandardScaler()
        lr = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=40
        )

        pipeline = Pipeline([
            ('scaler', preprocessor),
            ('regressor', lr)
        ])

        pipeline.fit(train_x, train_y)

        # Evaluate Metrics
        predicted_qualities = pipeline.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print out metrics
        print(f"XGBoost Regressor with Standardscaler (n_estimators={n_estimators:f}, learning_rate={learning_rate:f}, max_depth={max_depth:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # signature of model - for logging
        signature = infer_signature(train_x, lr.predict(train_x))
        print(signature)


        # listing all elements to be logged by mlflow

        # listing parameters and metrics to be logged: 
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # logging and registering the model (signature):
        mlflow.xgboost.log_model(lr, "xgboost_model", registered_model_name="xgboost_v.2.0.0_dataset_v.2.0.0", signature=signature)


        # logginig system-metrics: 
        mlflow.MlflowClient().get_run(run.info.run_id).data

        # logging the datasets: train and test
        mlflow.log_artifact("train_data.csv", artifact_path="datasets")
        mlflow.log_artifact("test_data.csv", artifact_path="datasets")

        os.remove("train_data.csv")
        os.remove("test_data.csv")


# listing hyperparameters you want to test 

train(200, 0.2, 10)
train(100, 0.1, 5)

   