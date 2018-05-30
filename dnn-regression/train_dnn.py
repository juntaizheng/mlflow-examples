import pandas
import tensorflow as tf
from tensorflow.estimator.inputs import numpy_input_fn

def train(args, pandasData):

	# Split data into a labels dataframe and a features dataframe
	labels = pandasData[args.label_col].values
	features = pandasData[args.feat_cols].values

	# Hold out test_percent of the data for testing.  We will use the rest for training.
	trainingFeatures, testFeatures, trainingLabels, testLabels = train_test_split(features, labels, test_size=args.test_percent)
	ntrain, ntest = len(trainingLabels), len(testLabels)
	print("Split data randomly into 2 sets: {} training and {} test instances.".format(ntrain, ntest))

	# Create input functions for both the training and testing sets.
	with tf.Session() as session:
		input_train = numpy_input_fn(trainingFeatures, trainingLabels, shuffle=True)
		input_test = numpy_input_fn(testFeatures, testLabels, shuffle=False)

	# Create TensorFlow columns based on passed in feature columns
	tf_feat_cols = []
	for col in args.feat_cols:
		tf_feat_cols.append(tf.feature_column.numeric_column(key=col))
	

