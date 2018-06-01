import pandas
import tensorflow as tf
from mlflow import log_metric, log_parameter, log_output_files, active_run_id
from mlflow.sklearn import log_model

def train(training_pandasData, label_col, feat_cols, hidden_units, steps, batch_size, training_data_path):

    print("training-data-path:    " + training_data_path)
    for hu in hidden_units:
        print("hidden-units:         ", hu)
    print("steps:                ", steps)
    print("batch_size:           ", batch_size)
    print("label-col:             " + label_col)
    for feat in feat_cols:
        print("feat-cols:             " + feat)
    #print('price' in training_pandasData.columns)
    # Split data into a labels dataframe and a features dataframe
    trainingLabels = training_pandasData[label_col].values
    print(trainingLabels)
    trainingFeatures = {}
    for feat in feat_cols:
        trainingFeatures[feat.replace(" ", "_")] = training_pandasData[feat].values
    #temp = training_pandasData.drop(label_col, axis=1)
    #input_train = tf.estimator.inputs.pandas_input_fn(temp, training_pandasData[label_col], shuffle=False)
    # Create input functions for both the training and testing sets.
    with tf.Session() as session:
        input_train = tf.estimator.inputs.numpy_input_fn(trainingFeatures, trainingLabels, shuffle=True, batch_size=batch_size)
        # input_test = numpy_input_fn(testFeatures, testLabels, shuffle=False)
    
    # Create TensorFlow columns based on passed in feature columns
    tf_feat_cols = []
    for col in feat_cols:
        tf_feat_cols.append(tf.feature_column.numeric_column(col.replace(" ", "_")))
    
    # Creating DNNRegressor
    regressor = tf.estimator.DNNRegressor(
        feature_columns=tf_feat_cols,
        hidden_units=hidden_units)

    # Training regressor on training input function
    regressor.train(
        input_fn=input_train,
        steps=steps)

    # Evaluating model on training data
    training_eval = regressor.evaluate(input_fn=input_train)

    training_rmse = training_eval["average_loss"]**0.5
    p = regressor.predict(input_fn=input_train)
    for prediction in p:
        print(prediction)
    print("Training RMSE:", training_rmse)

    log_parameter("Training data path", training_data_path)
    log_parameter("Label column", label_col)
    log_parameter("Feature columns", feat_cols)
    log_parameter("Hidden units", hidden_units)
    log_parameter("Steps", steps)
    log_parameter("Batch size", batch_size)
    log_parameter("Number of data points", len(training_pandasData[label_col].values))

    #Logging the rmse for both sets.
    log_metric("RMSE for training set", training_rmse)
    #log_metric("R2 score for test set", r2_score_test)

    log_output_files("outputs")

    #Saving the model as an artifact.
    #log_model(regressor, "model")

    #print("Model saved in mlruns/%s" % active_run_id())
