import pandas
import tensorflow as tf
from tensorflow.estimator.inputs import numpy_input_fn
from tensorflow.estimator import DNNRegressor

def train(training_pandasData, label_col, feat_cols, hidden_units, steps, training_data_path):

    print("training-data-path:    " + training_data_path)
    for hu in hidden_units:
        print("hidden-units:         ", hu)
    print("steps:                ", steps)
    print("label-col:             " + label_col)
    for feat in feat_cols:
        print("feat-cols:          " + feat)

    # Split data into a labels dataframe and a features dataframe
    trainingLabels = training_pandasData[label_col].values
    trainingfeatures = training_pandasData[feat_cols].values

    # Create input functions for both the training and testing sets.
    with tf.Session() as session:
        input_train = numpy_input_fn(trainingfeatures, trainingLabels)
        # input_test = numpy_input_fn(testFeatures, testLabels, shuffle=False)

    # Create TensorFlow columns based on passed in feature columns
    tf_feat_cols = []
    for col in feat_cols:
        tf_feat_cols.append(tf.feature_column.numeric_column(key=col))
    
    # Creating DNNRegressor
    regressor = DNNRegressor(
        feature_columns=tf_feat_cols,
        hidden_units=hidden_units)

    # Training regressor on training input function
    regressor.train(
        input_fn=input_train,
        steps = steps)

    # Evaluating model on training data
    training_eval = regressor.evaluate(input_fn=input_train)

    training_rmse = training_eval["average_loss"]**0.5

    log_parameter("Training data path", data_path)
    log_parameter("Label column", label_col)
    log_parameter("Feature columns", feat_cols)
    log_parameter("Hidden units", hidden_units)
    log_parameter("Steps", steps)
    log_parameter("Number of data points", len(features))

    #Logging the rmse for both sets.
    log_metric("RMSE for training set", training_rmse)
    #log_metric("R2 score for test set", r2_score_test)

    log_output_files("outputs")

    #Saving the model as an artifact.
    log_model(en, "model")

    print("Model saved in mlruns/%s" % active_run_id())