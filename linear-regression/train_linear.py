from time import time
from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import train_test_split
from mlflow import log_metric, log_parameter, log_output_files, active_run_id
from mlflow.sklearn import log_model

def train(args, pandasData):

    # Split data into a labels dataframe and a features dataframe
    labels = pandasData[args.label_col].values
    features = pandasData[args.feat_cols].values

    # Hold out test_percent of the data for testing.  We will use the rest for training.
    trainingFeatures, testFeatures, trainingLabels, testLabels = train_test_split(features, 
                                                                labels, test_size=args.test_percent)
    ntrain, ntest = len(trainingLabels), len(testLabels)
    print("Split data randomly into {} training and {} test instances.".format(ntrain, ntest))

    #We will use a linear Elastic Net model.
    en = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio)

    # Here we train the model and keep track of how long it takes.
    start_time = time()
    en.fit(trainingFeatures, trainingLabels)

    # Calculating the score of the model.
    r2_score_training = en.score(trainingFeatures, trainingLabels)
    r2_score_test = 0
    if args.test_percent != 0:
        r2_score_test = en.score(testFeatures, testLabels)
    timed = time() - start_time
    print("Training set score:", r2_score_training)
    if args.test_percent != 0:
        print("Test set score:", r2_score_test)

    #Logging the parameters for viewing later. Can be found in the folder mlruns/.
    if len(vars(args)) > 5:
        log_parameter("Data Path", args.data_path)
    log_parameter("Alpha", args.alpha)
    log_parameter("l1 ratio", args.l1_ratio)
    log_parameter("Testing set percentage", args.test_percent)
    log_parameter("Label column", args.label_col)
    log_parameter("Feature columns", args.feat_cols)
    log_parameter("Number of data points", len(features))

    #Logging the r2 score for both sets.
    log_metric("R2 score for training set", r2_score_training)
    if args.test_percent != 0:
        log_metric("R2 score for test set", r2_score_test)

    log_output_files("outputs")

    #Saving the model as an artifact.
    log_model(en, "model")

    print("Model saved in mlruns/%s" % active_run_id())

    #Determining how long the program took.
    print("This model took", timed, "seconds to train and test.")
    log_metric("Time to run", timed)
