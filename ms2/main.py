import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, CNN_2Test, CNN_3Test, CNN_1Test
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    xtrain, xtest, ytrain = load_data(args.data_path)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)


    if not args.test:
        from sklearn.model_selection import train_test_split
        xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, random_state=15)
    else : 
        xval, yval = xtest, None




    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)

    n_classes = get_n_classes(ytrain)
    if args.nn_type == "cnn" : 
        input_channels = 1  # Fashion MNIST is grayscale
        model = CNN(input_channels=input_channels, n_classes=n_classes)
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xval = xval.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
    if args.nn_type == "cnn2" : 
        input_channels = 1
        model = CNN_2Test(input_channels=input_channels, n_classes=n_classes)
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xval = xval.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)

    if args.nn_type == "cnn3" : 
        input_channels = 1
        model = CNN_3Test(input_channels=input_channels, n_classes=n_classes)
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xval = xval.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
        

    model.to(args.device)
    summary(model, input_size=(args.nn_batch_size, 1, 28, 28))

    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)


    preds_train = method_obj.fit(xtrain, ytrain)

    preds = method_obj.predict(xtest)

    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    if yval is not None: 
        val_preds = method_obj.predict(xval)
        acc = accuracy_fn(val_preds, yval)
        macrof1 = macrof1_fn(val_preds, yval)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)