import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        if k < 1 : 
            raise ValueError("K must be a strictly positive integer.")
        self.k = k
        self.task_kind =task_kind
        # init to None to handle error when calling predict before fit 
        self.training_data = None 
        self.training_labels = None 

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        self.training_data = training_data
        self.training_labels = training_labels
        pred_labels = self.predict(training_data)
        return pred_labels 

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        if self.training_data is None or self.training_labels is None : 
            raise ValueError("Can't predict with a model that has not been fitted! Call fit() before calling predict().")
        
        distances = np.sqrt(((test_data[:, np.newaxis] - self.training_data) ** 2).sum(axis=2)) # compute distances 

        nearest_neighbor_ids = np.argsort(distances, axis=1)[:, :self.k] # get k nearest neighbors 

        k_nearest_labels = self.training_labels[nearest_neighbor_ids] # get labels from nearest neighbors 

        test_labels = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), 1, k_nearest_labels) # get most common label among k nearest 
        return test_labels
