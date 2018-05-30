import argparse
from shutil import rmtree
from tempfile import mkdtemp
import train_linear
import utils

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

pandasData = utils.download_diamonds(temp_folder_path)
# Train the model based on the parameters provided.
train_linear.train(args, pandasData)

# Delete the temporary folder that stores the csv and parquet files.
rmtree(temp_folder_path)
