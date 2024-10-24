import numpy as np

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        self.mean = np.mean(training_data, 0)
        
        training_data_tilde = training_data - self.mean
        
        C = (training_data_tilde.T@training_data_tilde)/training_data_tilde.shape[0]
        
        eigvals, eigvecs = np.linalg.eigh(C)
        
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        
        self.W = eigvecs[:, 0:self.d]
        eg = eigvals[0:self.d]
        
        exvar = (np.sum(eg) / np.sum(eigvals)) * 100

        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        data_tilde = data - self.mean
        data_reduced = data_tilde @ self.W

        return data_reduced
        

