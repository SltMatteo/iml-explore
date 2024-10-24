import numpy as np
import sys
import os 




class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        # init to None to handle error when calling predict before fit 
        self.w = None

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """

        D = training_data.shape[1]
        I = np.eye(D) # for ridge 
        self.w = np.linalg.pinv(training_data.T @ training_data + self.lmda*I)@training_data.T@training_labels # closed form solution 
        #pred_regression_targets = training_data@self.w
        pred_regression_targets = self.predict(training_data)
        return pred_regression_targets

    
    def predict(self, test_data): 
         """
            Runs prediction on the test data. 

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns: 
                test_labels (np.array): labels of shape (N, regression_target_size)
         """
         if self.w is None :
             raise ValueError("Can't predict with a model that has not been fitted! Call fit() before calling predict().")
         
         pred_regression_targets = test_data@self.w 

         return pred_regression_targets
