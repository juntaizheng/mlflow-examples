import os
import pickle
import pandas
from mlflow.utils.file_utils import TempDir
from mlflow.projects import run
from mlflow import tracking
from mlflow.sklearn import load_pyfunc


def test_dnn():
    old_uri = tracking.get_tracking_uri()
    with TempDir(chdr=False, remove_on_exit=True) as tmp:
        try:
            diamonds = tmp.path("diamonds")
            artifacts = tmp.path("artifacts")
            os.mkdir(diamonds)
            os.mkdir(artifacts)
            tracking.set_tracking_uri(artifacts)
            # Download the diamonds dataset via mlflow run
            run(".", entry_point="download-example-data", version=None, parameters={"dir":diamonds}, 
            experiment_id=tracking._get_experiment_id(), mode="local", 
            cluster_spec=None, git_username=None, git_password=None, use_conda=True,
            use_temp_cwd=False, storage_dir=None)

            # Run the main dnn app via mlflow
            run(".", entry_point="linear-regression-main", version=None, 
            parameters={"training-data-path": os.path.join(diamonds, "train_diamonds.parquet"),
                        "test-data-path": os.path.join(diamonds, "test_diamonds.parquet"), 
                        "alpha": .001,
                        "l1-ratio": .5,
                        "label-col":"price"}, 
            experiment_id=tracking._get_experiment_id(), mode="local", 
            cluster_spec=None, git_username=None, git_password=None, use_conda=True,
            use_temp_cwd=False, storage_dir=None)

            # Loading the saved model as a pyfunc.
            initial = os.path.join(artifacts, os.listdir(artifacts)[0])

            print(os.listdir(initial)[0])

            pyfunc = load_pyfunc(os.path.join(initial, "model/model.pkl"))

            df = pandas.read_parquet(os.path.join(diamonds, "test_diamonds.parquet"))["price"]

            # Predicting from the saved pyfunc.

            predict = pyfunc.predict(df.values.reshape(-1, 1))

            print(type(predict))

            # Loading the saved predictions to compare to the pyfunc predictions.
            with open(os.path.join(artifacts, "predictions"), "rb") as f:
                saved = pickle.load(f)

            assert predict == saved
        finally:
            tracking.set_tracking_uri(old_uri)
