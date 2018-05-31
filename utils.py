import os
import urllib.request
import pandas
from pyspark import SparkContext
from pyspark.sql import SQLContext

# Downloads the diamonds dataset into the provided folder path.
def download_diamonds(temp_folder_path):
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
	pandasData['color'] = pandasData['color'].replace({'J':0, 'I':1, 
														'H':2, 'G':3, 'F':4, 'E':5, 'D':6})
	pandasData['clarity'] = pandasData['clarity'].replace({'I1':0, 
							'SI1':1, 'SI2':2, 'VS1':3, 'VS2':4, 'VVS1':5, 'VVS2':6, 'IF':7})

	return pandasData

def linear_arg_handler(args, pandasData):
	if len(vars(args)) > 5 or (len(vars(args)) == 4 and not args.feat_cols):
		print("data-path:    ", args.data_path)
	print("alpha:        ", args.alpha)
	print("l1-ratio:     ", args.l1_ratio)
	print("test-percent: ", args.test_percent)
	print("label-col:     " + args.label_col)

	# This is the case if the user specified which columns are to be feature columns.
	if args.feat_cols:
	    for col in args.feat_cols:
	        print("feat-cols:     " + col)
	# If no feature columns are specified, it is assumed all columns but the label are features.
	else:
	    print("No feature columns input. Assuming all columns but label column are features.")
	    args.feat_cols = []
	    for col in list(pandasData):
	        if not col == args.label_col:
	            print("feat-cols:     " + col)
	            args.feat_cols.append(col)
	return args
