# mlflow-examples
## [MLflow](http://mlflow.org) App Library
### gbt-regression
This app creates and fits an [XGBoost Gradient Boosted Tree](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) model based on parquet-formatted input data. The arguments to the program are as follows:
* `training-data-path`: (str, required) local path or URI of a parquet file containing training data
* `test-data-path`: (str, required) local path or URI of a parquet file containing test data
* `n-trees`: (int) number of trees for the regressor; default `100`
* `m-depth`: (int) maximum depth of trees of regressor; default `10`
* `learning-rate`: (float) learning rate of the model; ranges from `0.0` to `1.0`; default `.2`
* `loss`: (str) name of [loss function](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md) to be used; default `"rmse"`
* `label-col`: (str, required) name of label column in dataset; `string` input
* `feat-cols`: (str) names of columns in dataset to be used as features; input is one `string` with names delimited by commas
	* This argument is optional. If no argument is provided, it is assumed that all columns but the label column are feature columns.

This app currently assumes that the input data is all numerical.

The following commands should be run from the root repository directory:
>>>>>>> fbde21f653a2afe6552f3e361f37c1889b480c62

To run the app with default parameters while in the root directory, run the command 
```
mlflow run . -e gbt-regression-main -P data-path="insert/data/path/" -P label-col="insert.label.col" 
```

### linear-regression

This app creates and fits an [Elastic Net](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) model based on parquet-formatted input data. The arguments to the program are as follows:
* `training-data-path`: (str, required) local path or URI of a parquet file containing training data
* `test-data-path`: (str, required) local path or URI of a parquet file containing test data
* `alpha`: (float) alpha for the regressor; default `.001`
* `l1-ratio`: (float) l1 ratio to be used for the regressor; default `.5`
* `label-col`: (str, required) name of label column in dataset; `string` input
* `feat-cols`: (str) names of columns in dataset to be used as features; input is one `string` with names delimited by commas
    * This argument is optional. If no argument is provided, it is assumed that all columns but the label column are feature columns.

This app currently assumes that the input data is all numerical.

To run the app with default parameters while in the root directory, run the command 
```
mlflow run . -e linear-regression-main -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P label-col="insert.label.col"
```

### dnn-regression

This sample project creates and fits a Tensorflow [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor) model based on parquet-formatted input data. Then, the application exports the model to a local file and logs the model using MLflow's APIs. The arguments to the program are as follows:
* `model-dir`: (str, required) local path or URI of a directory to which the DNNRegressor's checkpoints and the final exported SavedModel will be written.
* `training-data-path`: (str, required) local path or URI of a parquet file containing training data
* `test-data-path`: (str, required) local path or URI of a parquet file containing test data
* `hidden-units`: (str, required) size and number of layers for the dnn; `string` input which has layers delimited by commas (i.e. "10,10" for two layers of 10 nodes each)
* `steps`: (int) steps to be run while training the regressor; default `100`
* `batch-size`: (int) batch size used for creation of input functions for training and evaluation; default `128`
* `label-col`: (str, required) name of label column in dataset
* `feat-cols`: (str) names of columns in dataset to be used as features; input is one `string` with names delimited by commas
    * This argument is optional. If no argument is provided, it is assumed that all columns but the label column are feature columns.

This app currently assumes that the input data is all numerical. Column names must adhere to TensorFlow [constraints](https://www.tensorflow.org/api_docs/python/tf/Operation#__init__).

To run the app with default parameters while in the root directory, run the command 
```
mlflow run . -e dnn-regression-main -P model-dir="insert/model/save/path" -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P hidden-units="10,10" -P label-col="insert.label.col"
```

### Downloading an Example Dataset

You can download example training & test parquet files containing the [diamonds](https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917/data-raw/diamonds.csv) dataset by running the command 
```
mlflow run  . -e download-example-data -P dest_dir="path/to/dir"
```
You can then use these files as data for running the example applications.

### Specifying Additional Parameters

To pass additional parameters to a `mlflow run` command, add `-P name-of-argument=value.of.argument` to the command. An example of adding custom parameters to the `gbt-regression` example app is as follows: 
```
mlflow run . -e gbt-regression-main -P data-path="insert/data/path/" -P label-col="insert.label.col" -P feat-cols="insert,feat,cols" -P n-trees=500
```

### Running MLflow from a Different Directory

To run an app from a different directory other than the root, replace the "." with the path to the folder containing the MLProject file. For example, the command to run the `linear-regression` app from `mlflow-examples`'s parent directory is
```
mlflow run mlflow-examples -e linear-regression-main -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P label-col="insert.label.col" 
```

### Running MLflow from a Git Repository

To run a MLflow project from a GitHub repository, replace the path to MlProject file folder with the SSH clone URI. For example, if you wanted to run the `dnn-regression` example application from a Git repository, run the command
```
mlflow run git@github.com:databricks/mlflow-examples.git -e dnn-regression-main -P model-dir="insert/model/save/path" -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P hidden-units="10,10" -P label-col="insert.label.col"
```

### linear-regression

This app creates and fits a Gradient Boosted Tree model based on parquet-formatted input data. The arguments to the program are as follows:
* `data-path`: local or absolute path to the parquet file to be fitted on; `string` input
* `alpha`: alpha for the regressor; default `.001`
* `l1-ratio`: l1 ratio to be used for the regressor; default `.5`
* `test-percent`: percentage of the input data that is held as the testing set; ranges from `0.0-1.0`; default `.3`
* `label-col`: name of label column in dataset; `string` input
* `feat-cols`: names of columns in dataset to be used as features; input is one `string` with names delimited by commas
    * This argument is optional. If no argument is provided, it is assumed that all columns but the label column are feature columns.

This app currently assumes that the input data is all numerical.

The following commands should be run from the root repository directory:

To run the app with default parameters on a dataset, run the code 
```
mlflow run . -e linear-regression-main -P data-path="insert/data/path/" -P label-col="insert.label.col"
```
where `insert/data/path/` is replaced with the actual path to the parquet data, `insert.label.col` is replaced with the label column.

To run an example of the app on the [diamonds dataset](https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917/data-raw/diamonds.csv), run the code 
```
mlflow run . -e linear-regression-example -P label-col="price"
```

To pass parameter values to the app, simply add `-P name-of-argument=value.of.argument` to the command. An example of passing custom parameters using the diamonds dataset is as follows: 
```
mlflow run . -e linear-regression-example -P alpha=.01 -P l1-ratio=.2 -P test-percent=.1 -P label-col="price" -P feat-cols="carat,cut,color,clarity,depth,table,x,y,z"
```

To run an app from a different directory, replace the "." with the path to the root repository folder. For example, the command to run the app on the diamonds dataset from the parent directory of `mlflow-examples` is:
```
mlflow run mlflow-examples -e linear-regression-example -P label-col="price" 
```

### deep neural network regression

This app creates and fits a Tensorflow DNNRegressor model based on parquet-formatted input data. The arguments to the program are as follows:
* `training-data-path`: local or absolute path to the parquet file to be fitted on; `string` input
* `test-data-path`: local or absolute path to the parquet file to be tested on; `string` input
* `hidden-units`: size and number of layers for the dnn; `string` input with layers delimited by commas (i.e. "10,10")
* `steps`: steps to be run whil training the regressor; default `100`
* `batch-size`: batch size used for creation of input functions for training and evaluation; default `128`
* `label-col`: name of label column in dataset; `string` input
* `feat-cols`: names of columns in dataset to be used as features; input is one `string` with names delimited by commas
    * This argument is optional. If no argument is provided, it is assumed that all columns but the label column are feature columns.

This app currently assumes that the input data is all numerical.

The following commands should be run from the root repository directory:

To run the app with default parameters on a dataset, run the code 
```
mlflow run . -e dnn-regression-main -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P hidden-units="10,10" -P label-col="insert.label.col"
```
where `insert/data/path/` is replaced with the actual path to the parquet data, `insert.label.col` is replaced with the label column.
