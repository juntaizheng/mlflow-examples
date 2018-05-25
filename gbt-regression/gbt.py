from mlflow import log_metric, log_parameter, log_output_files
import argparse
import pandas
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import os
import sys
from pathlib import Path
from sklearn import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import *
from sklearn.metrics import *
import xgboost as xgb
import time
from mlflow import log_metric, log_parameter, log_output_files, active_run_id
from mlflow.sklearn import log_model, save_model
#install pyarrow and pyspark and xgboost

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False
'''
Let’s start with an app to simplify training a Gradient Boosted Tree (GBT) classifier on a dataset. 
GBTs are an ensemble model frequently used to win ML competitions on Kaggle & popular for 
both classification & regression; however in our example we’ll focus on classification (discrete labels).

The app should accept as arguments:
* a parquet file
* a list of feature column names
* a label column name
* the number of trees to fit
* the max depth of each tree
* the loss function to use
* ADDED: the learning rate to use
* ADDED: the testing set size to use

Given these arguments, the app should train a single-machine scikit-learn GBT model on the data, 
and produce a pickled model file. The app should use MLflow tracking APIs to log the input parameters, 
the model file & the model’s training loss.

Specifically, you’ll need to write logic to:

* Wrap the notebook training code in a MLflow project with Python dependencies & parameterized entry points
* Convert the `diamonds` dataset from CSV to Parquet for testing (Sid can help with this)
* Parse parquet files into a Pandas DataFrame for the scikit-learn GBT model
* Pass parameters to the GBT classifier
* Save out the fitted model in pickled format

It’d be great to note any challenges / usability quirks you hit when using MLflow - 
we can use the feedback to refine the MLflow APIs.
'''
# An example local call: python example/experiments/gbt.py diamonds 100 10 .2 .3 rmse price "carat,cut,color,clarity,depth,table,x,y,z"
# An example mlflow call: mlflow run gbt-regression -P data_path="diamonds" -P n_trees=10 -P m_depth=10 -P learning_rate=.2 -P test_percent=.3 -P loss="rmse" -P label_col="price" -P feat_cols="carat","cut","color","clarity","depth","table","x","y","z"
# Parsing arguments.
parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="Path to parquet dataset file. Input 'diamonds' to use the sample dataset.",
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
args = parser.parse_args()

print("data_path:    ", args.data_path)
print("n_trees:      ", args.n_trees)
print("m_depth:      ", args.m_depth)
print("learning_rate:", args.learning_rate)
print("test_percent: ", args.test_percent)
print("loss:          " + args.loss)
print("label_col:     " + args.label_col)
for i in args.feat_cols:
	print("feat_cols      " + i)

# Conversion of CSV to Parquet. Only needed for testing the diamonds dataset.
# See http://blogs.quovantis.com/how-to-convert-csv-to-parquet-files/

#Checking if the diamonds parquet directory already exists. If not, we create it.
if args.data_path == "diamonds":
	if not Path(os.path.join(sys.path[0], 'diamonds_parquet')).exists():
		sc = SparkContext(appName="CSV2Parquet")
		sqlContext = SQLContext(sc)
		    
		schema = StructType([
		        StructField("carat", StringType(), True),
		        StructField("cut", StringType(), True),
		        StructField("color", StringType(), True),
		        StructField("clarity", StringType(), True),
		        StructField("depth", StringType(), True),
		        StructField("table", StringType(), True),
		        StructField("price", StringType(), True),
		        StructField("x", StringType(), True),
		        StructField("y", StringType(), True),
		        StructField("z", StringType(), True)])
		    
		rdd = sc.textFile(os.path.join(sys.path[0], 'diamonds.csv')).map(lambda line: 
				[float(i) if isfloat(i) else i for i in line.split(",")])
		df = sqlContext.createDataFrame(rdd, schema)
		df.write.parquet(os.path.join(sys.path[0], 'diamonds_parquet'))

	parquet_path = os.path.join(sys.path[0], 'diamonds_parquet')

	# Conversion of Parquet to pandas. See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_parquet.html
	pandasData = pandas.read_parquet(parquet_path)
	pandasData = pandasData.iloc[1:,:] # Remove top row. Hacky fix for diamonds csv to parquet conversion

	# Conversion of qualitative values to quantitative values. For diamonds only.
	pandasData['cut'] = pandasData['cut'].replace({'"Fair"':0, '"Good"':1, '"Very Good"':2, '"Premium"':3, '"Ideal"':4})
	pandasData['color'] = pandasData['color'].replace({'"J"':0, '"I"':1, '"H"':2, '"G"':3, '"F"':4, '"E"':5, '"D"':6})
	pandasData['clarity'] = pandasData['clarity'].replace({'"I1"':0, '"SI1"':1, '"SI2"':2, '"VS1"':3, '"VS2"':4, 
															'"VVS1"':5, '"VVS2"':6, '"IF"':7})
else:
	pandasData = pandas.read_parquet(args.data_path)
# Split data into a labels dataframe and a features dataframe
labels = pandasData[args.label_col].values
featureNames = args.feat_cols
features = pandasData[featureNames].values
# Normalize features (columns) to have unit variance
features = normalize(features, axis=0)

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
accuracy = xgbr.score(testFeatures, testLabels)
time = time.time() - start_time
print(accuracy)

#Logging the parameters for viewing later. Can be found in the folder mlruns/.
log_parameter("Data Path", args.data_path)
log_parameter("Number of trees", args.n_trees)
log_parameter("Max depth of trees", args.m_depth)
log_parameter("Learning rate", args.learning_rate)
log_parameter("Testing set percentage", args.test_percent)
log_parameter("Loss function used", args.loss)
log_parameter("Label column", args.label_col)
log_parameter("Feature columns", args.feat_cols)
log_parameter("Number of data points", len(features))

#Logging the accuracy.
log_metric("Accuracy", accuracy)

log_output_files("outputs")

#Saving the model as an artifact.
log_model(xgbr, "model")

print("Model saved in mlruns/%s" % active_run_id())

#Determining how long the program took.
print("This model took", time, "seconds to run")
log_metric("Time to run", time)