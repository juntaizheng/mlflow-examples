# [MLflow](http://mlflow.org) App Library

Collection of pluggable MLflow apps (MLflow projects). You can call the apps in this repository to:
* Seamlessly embed ML functionality into your own applications
* Reproducibly train models from a variety of frameworks on big & small data, without worrying about installing dependencies

## Getting Started
### Running Apps via the CLI
Let's start by running the DNNRegressor app, which trains a deep feedforward neural net using TensorFlow.

First, download example training & test parquet files by running:
TODO: Make this example work more seamlessly out of the box (use bash commands to create a temporary dir for downloading the data for the user, etc).
 
```
mlflow run  . -e download-example-data -P dest_dir="path/to/dir"
```

This will download the diamonds [diamonds](https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917/data-raw/diamonds.csv) dataset to the specified path.

Then, train a neural network on the data, saving the fitted network as an MLflow model. See the app docs (TODO link to app docs) for more info on available parameters
```
mlflow run examples/dnn-regression/ -e main -P model-dir="insert/model/save/path" -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P hidden-units="10,10" -P label-col="insert.label.col"
```

We can now use the fitted model to make predictions on our test data via the MLflow CLI:
```
TODO: Add command here
```

### Calling an App in Your Code

TODO: we should write/include an example notebook that shows how to call an app via the MLflow Python API

Calling an app from your code is simple  - just use MLflow's [Python API](https://mlflow.org/docs/latest/projects.html#building-multi-step-workflows):
```
# Train a TensorFlow DNNRegressor, exporting it as an MLflow model
train_data_path = "..."
test_data_path = "..."
label_col = "..."
exported_model_path = "..."
hidden_units = [10, 10]
mlflow.projects.run(uri="git@github.com:databricks/mlflow-examples.git#examples/dnn-regression/", parameters=[("model-dir", exported_model_path), ("training-data-path", train_data_path), ("test-data-path", test_data_path), ("hidden-units", ",".join(map(str, hidden_units)), ("label-col", label_col)])
```

## Apps

The library contains the following apps:

### dnn-regression

This app creates and fits a Tensorflow [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor) model based on parquet-formatted input data. Then, the application exports the model to a local file and logs the model using MLflow's APIs. See more info [here](examples/dnn-regression/).

### gbt-regression
This app creates and fits an [XGBoost Gradient Boosted Tree](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) model based on parquet-formatted input data. See more info [here](examples/gbt-regression/).

### linear-regression

This app creates and fits an [Elastic Net](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) model based on parquet-formatted input data. See more info [here](examples/linear-regression/).

## Contributing

If you would like to contribute to this library, please see the [contribution guide](CONTRIBUTING.md) for details.
