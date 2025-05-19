import logging
import warnings
import dagshub

#set up connection to dagshub for mlflow
dagshub.init(repo_owner='fienme', repo_name='Modern-Data-Analytics', mlflow=True)

# model training with mlflow logging
def train(it_n_estimators, it_learning_rate, it_max_depth):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from xgboost import XGBRegressor
    from urllib.parse import urlparse
    import json

    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    from mlflow.models import infer_signature

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

    # logging list with features in json file
    target_column = "quality"
    features = [col for col in data.columns if col != target_column]
    print("Feature names:", features)

    with open("features.json", "w") as f:
        json.dump(features, f)

    # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run():
        # Execute ElasticNet
        lr = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=40
        )
        lr.fit(train_x, train_y)

        # Evaluate Metrics
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print out metrics
        print(f"XGBoost Regressor (n_estimators={n_estimators:f}, learning_rate={learning_rate:f}, max_depth={max_depth:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # Log parameter, metrics, and model to MLflow
        mlflow.sklearn.autolog()
        mlflow.xgboost.autolog()

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        #  Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")





train(150, 0.15, 2)
train(200, 0.2, 10)
train(100, 0.1, 5)