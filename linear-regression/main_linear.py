import argparse
import pandas
import train_linear

# Trains a single-machine scikit-learn Elastic Net model on the provided data file, 
# producing a pickled model file. Uses MLflow tracking APIs to log the input parameters, 
# the model file, and the model's training loss.

#Parsing arguments.
parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="Path to parquet dataset file.",
                    type=str)
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

print("data-path:    ", args.data_path)
print("alpha:        ", args.alpha)
print("l1-ratio:     ", args.l1_ratio)
print("test-percent: ", args.test_percent)
print("label-col:     " + args.label_col)
for i in args.feat_cols:
    print("feat-cols      " + i)

#Reading the parquet file into a pandas dataframe.
pandasData = pandas.read_parquet(args.data_path)

# Train the model based on the parameters provided.
train_linear.train(args, pandasData)
