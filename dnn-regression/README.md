# [MLflow](http://mlflow.org) Deep Neural Network Regression

This sample project creates and fits a Tensorflow [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor) model based on parquet-formatted input data. Then, the application exports the model to a local file and logs the model using MLflow's APIs. The arguments to the program are as follows:
* `training-data-path`: local or absolute path to the parquet file to be fitted on; `string` input
* `test-data-path`: local or absolute path to the parquet file to be tested on; `string` input
* `hidden-units`: size and number of layers for the dnn; `string` input with layers delimited by commas (i.e. "10,10" for two layers of 10 nodes each)
* `steps`: steps to be run whil training the regressor; default `100`
* `batch-size`: batch size used for creation of input functions for training and evaluation; default `128`
* `label-col`: name of label column in dataset; `string` input
* `feat-cols`: names of columns in dataset to be used as features; input is one `string` with names delimited by commas
    * This argument is optional. If no argument is provided, it is assumed that all columns but the label column are feature columns.

This example code currently only works for numerical data. In addition, column names must adhere to TensorFlow [constraints](https://www.tensorflow.org/api_docs/python/tf/Operation#__init__).

To run the project locally with default parameters on a dataset while in the parent directory, run the command
```
mlflow run dnn-regressor -e main -P model-dir="insert/model/save/path" -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P hidden-units="10,10" -P label-col="insert.label.col"
```
where `insert/model/save/path` is replaced with the path to where you want to export `insert/data/path/` is replaced with the actual path to the parquet data, and `insert.label.col` is replaced with the label column.

You can download example data from the [diamonds](https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917/data-raw/diamonds.csv) dataset by running the command
```
mlflow run dnn-regressor -e download-example-data -P dir="insert/download/path"
```
where `insert/download/path` is replaced by the directory you want to save the resulting training and test parquet data files. You can then use these files as data for running the example application.

To run the project from a git repository, run the command
```
mlflow run git@github.com:databricks/dnn-regressor.git -v master -e main -P model-dir="insert/model/save/path" -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P hidden-units="10,10" -P label-col="insert.label.col"
```
