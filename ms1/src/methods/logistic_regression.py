import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500, task_kind="classification"):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.task_kind = task_kind
        self.w = None

    def f_softmax(self, data, weights):
        """
        Softmax function for multi-class logistic regression.

        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        e = data @ weights
        e_max = np.max(e, axis=-1, keepdims=True)
        exp_x = np.exp(e - e_max)
        sum_exp = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / sum_exp

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        D = training_data.shape[1]  # number of features
        C = get_n_classes(training_labels)  # number of classes
        onehot = label_to_onehot(training_labels)  # one-hot encoding of the labels

        # weights initialization 
        weights = np.random.normal(-0.05, 0.05, (D, C))

        # gradient descent 
        for _ in range(self.max_iters):
            y = self.f_softmax(training_data, weights)
            gradient = training_data.T @ (y - onehot)
            weights = weights - self.lr * gradient

        self.w = weights
        predictions = self.f_softmax(training_data, weights)
        return np.argmax(predictions, axis=-1)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        # Check if the model has been trained
        if self.w is None:
            raise ValueError(
                "Can't do a prediction with a model that has not been trained ! Please call fit() before predict().")

        pred_labels = self.f_softmax(test_data, self.w)
        return np.argmax(pred_labels, axis=-1)
