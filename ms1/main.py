import argparse

import numpy as np

import time

from src.data import load_data
from src.methods.dummy import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression 
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os
np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
        of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors

    ##EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('features.npz',allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest =feature_data['xtrain'],feature_data['xtest'],\
        feature_data['ytrain'],feature_data['ytest'],feature_data['ctrain'],feature_data['ctest']

    ##ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path,'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    if not args.test:
        nbr_samples = xtrain.shape[0]
        fraction_train = 0.8  # 80% for training and 20% for testing
        random_perm = np.random.permutation(nbr_samples)  # shuffle the indices

        nbr_train = int(nbr_samples * fraction_train)
        xtest = xtrain[random_perm[nbr_train:]] 
        ytest = ytrain[random_perm[nbr_train:]]
        ctest = ctrain[random_perm[nbr_train:]]
        xtrain = xtrain[random_perm[:nbr_train]] 
        ytrain = ytrain[random_perm[:nbr_train]]
        ctrain = ctrain[random_perm[:nbr_train]]

    if args.normalization_method == "gaussian" : 
        xtrain_mean = np.mean(xtrain) 
        xtrain_std = np.std(xtrain)
        ctrain_mean = np.mean(ctrain)
        ctrain_std = np.std(ctrain) 

        normalize_fn(xtrain, xtrain_mean, xtrain_std) 
        normalize_fn(xtest, xtrain_mean, xtrain_std) 
        normalize_fn(ctrain, ctrain_mean, ctrain_std) 
        normalize_fn(ctest, ctrain_mean, ctrain_std) 

    if args.normalization_method == "minmax" : 
        xmax = np.max(xtrain, axis=0) 
        xmin = np.min(xtrain, axis=0)
        cmax = np.max(ctrain, axis=0) 
        cmin = np.min(ctrain, axis=0) 


        xtrain = (xtrain - xmax) / (xmax - xmin) 
        xtest = (xtest - xmax) / (xmax - xmin) 
        ctrain = (ctrain - cmax) / (cmax - cmin) 
        ctest = (ctest - cmax) / (cmax - cmin) 
        
        
    if args.method == "linear_regression" : 
        append_bias_term(ctrain) 
        append_bias_term(ctest) 
    if args.method == "logistic_regression":
        append_bias_term(xtrain) 
        append_bias_term(xtest)


    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "linear_regression": 
        method_obj = LinearRegression(args.lmda)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(args.lr, args.max_iters)
    
    elif args.method == "knn":
        if args.task == "center_locating":
            method_obj = KNN(args.K, "regression")
        else:
            method_obj = KNN(args.K, "classification")


    if args.task == "center_locating":
        s1 = time.time()
        # Fit parameters on training data
        preds_train = method_obj.fit(xtrain, ctrain)

        # Perform inference for training and test data
        train_pred = method_obj.predict(xtrain)
        preds = method_obj.predict(xtest)

        s2 = time.time()
        diff = s2-s1
        print(f"\n{args.method} takes: {diff:.10f} seconds")

        ## Report results: performance on train and valid/test sets
    
        train_loss = mse_fn(train_pred, ctrain)
        loss = mse_fn(preds, ctest)

        print(f"\nTrain loss = {train_loss:.3f}% - Test loss = {loss:.3f}")

    elif args.task == "breed_identifying":

        s1 = time.time()

        # Fit (:=train) the method on the training data for classification task
        preds_train = method_obj.fit(xtrain, ytrain)

        # Predict on unseen data
        preds = method_obj.predict(xtest)

        s2 = time.time()
        diff = s2-s1
        print(f"\n{args.method} takes: {diff:.10f} seconds")

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    else:
        raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")

    parser.add_argument('--normalization_method', default="none", type=str, help="the method with which the data gets normalized")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
