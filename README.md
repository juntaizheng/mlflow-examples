# mlflow-examples
## [MLflow](http://mlflow.org) App Library

This example library contains projects demonstrating usage of different model types supported by MLflow.

### Downloading an Example Dataset

You can download example training & test parquet files containing the [diamonds](https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917/data-raw/diamonds.csv) dataset by running the command 
```
mlflow run  . -e download-example-data -P dest_dir="path/to/dir"
```
You can then use these files as data for running the example applications.

### Specifying Additional Parameters

To pass additional parameters to a `mlflow run` command, add `-P name-of-argument=value.of.argument` to the command. An example of adding custom parameters to the `gbt-regression` example app is as follows: 
```
mlflow run examples/gbt-regression/ -e main -P data-path="insert/data/path/" -P label-col="insert.label.col" -P feat-cols="insert,feat,cols" -P n-trees=500
```

### Running MLflow from Code

You can use MLflow's [Python API](https://mlflow.org/docs/latest/projects.html#building-multi-step-workflows) to run a Mlflow project in your own code. For example, running an app from this library's Git repo using the API would look like the following:
```
mlflow.projects.run(uri="git@github.com:databricks/mlflow-examples.git#examples/dnn-regression/", parameters=[("model-dir","insert/model/save/path"), ("training-data-path","insert/data/path/"), ("test-data-path","insert/data/path/"), ("hidden-units","10,10"), ("label-col","insert.label.col")])
```

## Apps

### dnn-regression

This app creates and fits a Tensorflow [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor) model based on parquet-formatted input data. Then, the application exports the model to a local file and logs the model using MLflow's APIs. See more info [here](examples/dnn-regression/).

### gbt-regression
This app creates and fits an [XGBoost Gradient Boosted Tree](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) model based on parquet-formatted input data. See more info [here](examples/gbt-regression/).

### linear-regression

This app creates and fits an [Elastic Net](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) model based on parquet-formatted input data. See more info [here](examples/linear-regression/).

## Contributing

If you would like to contribute to this library, please see the [contribution guide](CONTRIBUTING.md) for details.
