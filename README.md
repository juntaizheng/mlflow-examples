# mlflow-examples
## [MLflow](http://mlflow.org) App Library

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

### Running MLflow from a Git Repository

To run a MLflow project from a GitHub repository, replace the path to MlProject file folder with the SSH or HTTPS clone URI. To specify a project within a subdirectory, add a '#' to the end of the URI argument, followed by the path from the repository's root directory to the subdirectory containing the desired project. For example, if you wanted to run the `dnn-regression` example application from a Git repository, run the command
```
mlflow run git@github.com:databricks/mlflow-examples.git#examples/dnn-regression/ -e main -P model-dir="insert/model/save/path" -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P hidden-units="10,10" -P label-col="insert.label.col"
```

### dnn-regression

This sample project creates and fits a Tensorflow [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor) model based on parquet-formatted input data. Then, the application exports the model to a local file and logs the model using MLflow's APIs. See more info [here](examples/dnn-regression/).

### gbt-regression
This app creates and fits an [XGBoost Gradient Boosted Tree](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) model based on parquet-formatted input data. See more info [here](examples/gbt-regression/).

### linear-regression

This app creates and fits an [Elastic Net](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) model based on parquet-formatted input data. See more info [here](examples/linear-regression/).
