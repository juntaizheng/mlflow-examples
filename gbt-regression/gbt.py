import argparse
import csv
import os
import pandas
import sys
import time
import urllib.request
import xgboost as xgb

from pathlib import Path

from pyspark import SparkContext, SparkFiles
from pyspark.sql import SQLContext
from pyspark.sql.types import *

from sklearn import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import *
from sklearn.metrics import *

from mlflow import log_metric, log_parameter, log_output_files, active_run_id
from mlflow.sklearn import log_model, save_model

"""Trains a single-machine scikit-learn GBT model on the provided data file, producing a pickled model file. 
Uses MLflow tracking APIs to log the input parameters, the model file, and the model's training loss."""

# An example local call: 
# python gbt-regression/gbt.py diamonds 100 10 .2 .3 rmse price "carat,cut,color,clarity,depth,table,x,y,z" 1
# An example mlflow call: 
# mlflow run gbt-regression -e example -P data-path="diamonds" -P n-trees=100 -P m-depth=10 -P learning-rate=.2 -P test-percent=.3 -P loss="rmse" -P label-col="price" -P feat-cols="carat","cut","color","clarity","depth","table","x","y","z"

def isfloat(value):
	try:
		float(value)
		return True
	except ValueError:
		return False

# Parsing arguments.
parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="Path to parquet dataset file.",
                    type=str)
parser.add_argument("n_trees", help="Number of trees to fit.",
                    type=int)
parser.add_argument("m_depth", help="Max depth of trees.",
                    type=int)
parser.add_argument("learning_rate", help="Learning rate of the model.",
					type=float)
parser.add_argument("test_percent", help="Percent of data to use as test data.",
					type=float)
parser.add_argument("loss", help="""Loss function to use. See 
								https://github.com/dmlc/xgboost/blob/master/doc/parameter.md for list of functions.""",
                    type=str)
parser.add_argument("label_col", help="Name of label column.",
                    type=str)
parser.add_argument("feat_cols", help="List of feature column names. Input must be a single string with columns delimited by commas.",
                    type=lambda s: [str(i) for i in s.split(',')])
parser.add_argument("example", help="""Input 1 if you want to run the diamonds example, 0 if you are using your own data.
								The data-path argument will be ignored if this value is 1.""",
					type=int)

args = parser.parse_args()

print("data-path:    ", args.data_path)
print("n-trees:      ", args.n_trees)
print("m-depth:      ", args.m_depth)
print("learning-rate:", args.learning_rate)
print("test-percent: ", args.test_percent)
print("loss:          " + args.loss)
print("label-col:     " + args.label_col)
for i in args.feat_cols:
	print("feat-cols      " + i)
print("use example?  ", args.example)

# Conversion of CSV to Parquet. Only needed for testing the diamonds dataset.
# See http://blogs.quovantis.com/how-to-convert-csv-to-parquet-files/

#Checking if the diamonds parquet directory already exists. If not, we create it.
if args.example == 1:
	if not Path(os.path.join(sys.path[0], 'diamonds_parquet')).exists():
		print("Creating diamonds dataset parquet file...")
		url = "https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv"
		sc = SparkContext(appName="CSV2Parquet")
		sqlContext = SQLContext(sc)
		    
		# schema = StructType([
		#         StructField("carat", StringType(), True),
		#         StructField("cut", StringType(), True),
		#         StructField("color", StringType(), True),
		#         StructField("clarity", StringType(), True),
		#         StructField("depth", StringType(), True),
		#         StructField("table", StringType(), True),
		#         StructField("price", StringType(), True),
		#         StructField("x", StringType(), True),
		#         StructField("y", StringType(), True),
		#         StructField("z", StringType(), True)])
		    
		# rdd = sc.textFile(os.path.join(sys.path[0], 'diamonds.csv')).map(lambda line: 
		# 		[float(i) if isfloat(i) else i for i in line.split(",")])
		# df = sqlContext.createDataFrame(rdd, schema)
		if not Path("diamonds.csv").is_file():
			urllib.request.urlretrieve(url, "diamonds.csv")
		df = sqlContext.read.format("csv").option("header", "true").load("diamonds.csv")
		df.write.parquet(os.path.join(sys.path[0], 'diamonds_parquet'))
		print("Diamonds dataset parquet file created.")

	parquet_path = os.path.join(sys.path[0], 'diamonds_parquet')

	# Conversion of Parquet to pandas. See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_parquet.html
	pandasData = pandas.read_parquet(parquet_path)
	# Conversion of qualitative values to quantitative values. For diamonds only.
	pandasData['cut'] = pandasData['cut'].replace({'Fair':0, 'Good':1, 'Very Good':2, 'Premium':3, 'Ideal':4})
	pandasData['color'] = pandasData['color'].replace({'J':0, 'I':1, 'H':2, 'G':3, 'F':4, 'E':5, 'D':6})
	pandasData['clarity'] = pandasData['clarity'].replace({'I1':0, 'SI1':1, 'SI2':2, 'VS1':3, 'VS2':4, 
															'VVS1':5, 'VVS2':6, 'IF':7})
else:
	pandasData = pandas.read_parquet(args.data_path)
# Split data into a labels dataframe and a features dataframe
labels = pandasData[args.label_col].values
featureNames = args.feat_cols
features = pandasData[featureNames].values

# Hold out test_percent of the data for testing.  We will use the rest for training.
trainingFeatures, testFeatures, trainingLabels, testLabels = train_test_split(features, labels, test_size=args.test_percent)
ntrain, ntest = len(trainingLabels), len(testLabels)
print("Split data randomly into 2 sets: {} training and {} test instances.".format(ntrain, ntest))

# We will use a GBT regressor model.
xgbr = xgb.XGBRegressor(max_depth = args.m_depth, learning_rate = args.learning_rate, n_estimators = args.n_trees)

# Here we train the model and keep track of how long it takes.
start_time = time.time()
xgbr.fit(trainingFeatures, trainingLabels, eval_metric = args.loss)

# Calculating the score of the model.
r2_score_training = xgbr.score(trainingFeatures, trainingLabels)
r2_score_test = xgbr.score(testFeatures, testLabels)
time = time.time() - start_time
print("Training set score:", r2_score_training)
print("Test set score:", r2_score_test)

#Logging the parameters for viewing later. Can be found in the folder mlruns/.
if args.example == 0:
	log_parameter("Data Path", args.data_path)
else:
	log_parameter("Data Path", parquet_path)
log_parameter("Number of trees", args.n_trees)
log_parameter("Max depth of trees", args.m_depth)
log_parameter("Learning rate", args.learning_rate)
log_parameter("Testing set percentage", args.test_percent)
log_parameter("Loss function used", args.loss)
log_parameter("Label column", args.label_col)
log_parameter("Feature columns", args.feat_cols)
log_parameter("Number of data points", len(features))

#Logging the r2 score for both sets.
log_metric("R2 score for training set", r2_score_training)
log_metric("R2 score for test set", r2_score_test)

log_output_files("outputs")

#Saving the model as an artifact.
log_model(xgbr, "model")

print("Model saved in mlruns/%s" % active_run_id())

#Determining how long the program took.
print("This model took", time, "seconds to train and test.")
log_metric("Time to run", time)