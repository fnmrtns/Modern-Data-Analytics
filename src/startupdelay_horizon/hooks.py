from kedro.framework.hooks import hook_impl
from kedro_mlflow.framework.hooks import MlflowHook
from pyspark.sql import SparkSession
from pyspark import SparkConf
import mlflow
import os


class SparkHooks:
    """Configure Spark session after Kedro context is created."""

    @hook_impl
    def after_context_created(self, context):
        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())
        spark = (
            SparkSession.builder.appName(context.project_path.name)
            .enableHiveSupport()
            .config(conf=spark_conf)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")


class PlainMlflowHook(MlflowHook):
    """
    Set DagsHub tracking URI using environment variables (no dotenv).
    Requires MLFLOW_TRACKING_USERNAME and optionally MLFLOW_TRACKING_PASSWORD or token.
    """

    @hook_impl
    def after_context_created(self, context):
        mlflow.set_tracking_uri("https://dagshub.com/fienme/Modern-Data-Analytics.mlflow")
        # Do not call super() unless you rely on additional MlflowHook logic
        # super().after_context_created(context)


class TagAndParamHook:
    """
    Add MLflow tags for filtering.
    """

    @hook_impl
    def before_pipeline_run(self, run_params):
        mlflow.set_experiment("kedro-startupdelay")
        mlflow.set_tag("experiment", "baseline")
        mlflow.set_tag("run_type", "catboost")
        mlflow.set_tag("owner", "tanguy")
        mlflow.set_tag("pipeline", run_params.get("pipeline_name", "__default__"))
        mlflow.set_tag("run_id", run_params["run_id"])


# Register the hooks
mlflow_hook = PlainMlflowHook()
HOOKS = (SparkHooks(), mlflow_hook, TagAndParamHook())
