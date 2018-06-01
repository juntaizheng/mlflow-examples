import argparse
import pandas
import train_dnn
import utils

# Trains a single-machine Tensorflow DNNRegressor model on the provided data file, 
# producing a pickled model file. Uses MLflow tracking APIs to log the input parameters, 
# the model file, and the model's training loss.

#Parsing arguments.
parser = argparse.ArgumentParser()
parser.add_argument("training_data_path", help="Path to training parquet dataset file.",
                    type=str)
parser.add_argument("hidden_units", help="Hidden layer dimensions for the model.",
                    type=lambda s: [str(i) for i in s.split(',')])
parser.add_argument("steps", help="Number of steps for the training of the model to take.",
                    type=int)
parser.add_argument("batch_size", help="Number of steps for the training of the model to take.",
                    type=int)
parser.add_argument("label_col", help="Name of label column.",
                    type=str)
parser.add_argument("--feat-cols", help="List of feature column names. "
                        "Input must be a single string with columns delimited by commas.",
                    type=lambda s: [str(i) for i in s.split(',')])

args = parser.parse_args()

# Reading the parquet file into a pandas dataframe.
pandasData = pandas.read_parquet(args.training_data_path)

# Handle determining feature columns.
feat_cols = utils.get_feature_cols(args.feat_cols, args.label_col, list(pandasData))

# For testing with diamonds dataset.
# pandasData['cut'] = pandasData['cut'].replace({'Fair':0, 'Good':1, 
#                                                     'Very Good':2, 'Premium':3, 'Ideal':4})
# pandasData['color'] = pandasData['color'].replace({'J':0, 'I':1, 
#                                                         'H':2, 'G':3, 'F':4, 'E':5, 'D':6})
# pandasData['clarity'] = pandasData['clarity'].replace({'I1':0, 
#                             'SI1':1, 'SI2':2, 'VS1':3, 'VS2':4, 'VVS1':5, 'VVS2':6, 'IF':7})

# pandasData.apply(pandas.to_numeric, errors='ignore')
# pandasData = pandasData.astype('float64')
# print(pandasData.dtypes)
hidden_units = []
for hu in args.hidden_units:
    hidden_units.append(int(hu))

# Train the model based on the parameters provided.
train_dnn.train(pandasData, args.label_col, feat_cols, 
                    hidden_units, args.steps, args.batch_size, args.training_data_path)
