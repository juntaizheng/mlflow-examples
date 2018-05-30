import argparse
import os
import urllib.request
from shutil import rmtree
from tempfile import mkdtemp
import pandas
from pyspark import SparkContext
from pyspark.sql import SQLContext
import train_linear

# Trains a single-machine scikit-learn Elastic Net model on the provided data file, 
# producing a pickled model file. Uses MLflow tracking APIs to log the input parameters,
# the model file, and the model's training loss.

#Parsing arguments.
parser = argparse.ArgumentParser()

parser.add_argument("alpha", help="Alpha value for Elastic Net linear regressor.",
                    type=float)
parser.add_argument("l1_ratio", help="""L1 ratio for Elastic Net linear regressor. 
                        See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
                        for more details.""",
                    type=float)
parser.add_argument("test_percent", help="Percent of data to use as test data.",
                    type=float)
parser.add_argument("label_col", help="Name of label column.",
                    type=str)
parser.add_argument("feat_cols", help="""List of feature column names. 
                        Input must be a single string with columns delimited by commas.""",
                    type=lambda s: [str(i) for i in s.split(',')])

args = parser.parse_args()

print("alpha:        ", args.alpha)
print("l1-ratio:     ", args.l1_ratio)
print("test-percent: ", args.test_percent)
print("label-col:     " + args.label_col)
for i in args.feat_cols:
    print("feat-cols      " + i)

# Conversion of CSV to Parquet. Only needed for testing the diamonds dataset.

# Creating a temporary directory for the storage of the csv and parquet file. 
# Will be deleted at the end of the script.
temp_folder_path = mkdtemp()
sc = SparkContext(appName="CSV2Parquet")
sqlContext = SQLContext(sc)

#Downloading csv file from ggplot2's hosted dataset on github.
url = "https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917/data-raw/diamonds.csv"
print("Downloading diamonds csv file...")
urllib.request.urlretrieve(url, os.path.join(temp_folder_path, "diamonds.csv"))
df = sqlContext.read.format("csv").option("header", "true")
df = df.load(os.path.join(temp_folder_path, "diamonds.csv"))
print("Downloaded diamonds csv file.")
print("Creating diamonds dataset parquet file...")
df.write.parquet(os.path.join(temp_folder_path, "diamonds_parquet"))
print("Diamonds dataset parquet file created.")

parquet_path = os.path.join(temp_folder_path, "diamonds_parquet")

# Conversion of Parquet to pandas. 
# See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_parquet.html
pandasData = pandas.read_parquet(parquet_path)

# Conversion of qualitative values to quantitative values. For diamonds only.
pandasData['cut'] = pandasData['cut'].replace({'Fair':0, 'Good':1, 
                                                'Very Good':2, 'Premium':3, 'Ideal':4})
pandasData['color'] = pandasData['color'].replace({'J':0, 'I':1, 'H':2, 'G':3, 'F':4, 'E':5, 'D':6})
pandasData['clarity'] = pandasData['clarity'].replace({'I1':0, 'SI1':1, 'SI2':2, 
                                                        'VS1':3, 'VS2':4, 'VVS1':5, 'VVS2':6, 'IF':7})
# Train the model based on the parameters provided.
train_linear.train(args, pandasData)

# Delete the temporary folder that stores the csv and parquet files.
rmtree(temp_folder_path)
