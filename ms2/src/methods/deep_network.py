import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, hidden_dimensions=(3000, 200, 1000, 200)):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_dimensions[0])
        self.bn1 = nn.BatchNorm1d(hidden_dimensions[0])
        self.fc2 = nn.Linear(hidden_dimensions[0], hidden_dimensions[1])
        self.bn2 = nn.BatchNorm1d(hidden_dimensions[1])
        self.fc3 = nn.Linear(hidden_dimensions[1], hidden_dimensions[2])
        self.bn3 = nn.BatchNorm1d(hidden_dimensions[2])
        self.fc4 = nn.Linear(hidden_dimensions[2], hidden_dimensions[3])
        self.bn4 = nn.BatchNorm1d(hidden_dimensions[3])
        self.fc5 = nn.Linear(hidden_dimensions[3], n_classes)
        self.dropout = nn.Dropout(0.5)

        # Initialize weights using He initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        # chose (or test to find optimal) values for these parameters
        output_channels = 10 
        conv_kernel_size = 3 
        stride = 1 
        padding = 1
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=conv_kernel_size, stride=stride, padding=padding)
        # again, chose these wisely 
        pool_kernel_size = 2
        pool_stride = 2
        pool_padding = 0
        self.maxpool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        # and again 
        H, W = 28, 28 # dimensions of the images of the dataset (https://www.kaggle.com/datasets/zalando-research/fashionmnist). 
                      #I specify W because its cleaner to look at but one could just declare res = 28 since the images are square anyways 
        conv_output_size = ((H - conv_kernel_size + 2 * padding) // stride + 1)
        pool_output_size = ((conv_output_size - pool_kernel_size + 2 * pool_padding) // pool_stride + 1)
        output_dim = output_channels * (pool_output_size**2) 
        self.out = nn.Linear(output_dim, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = self.maxpool(F.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        preds = self.out(x) 
        return preds
    

class CNN_2Test(nn.Module):
    """
    This class was me testing different architectures. 

    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # added rn
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # added rn 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  
        self.dropout = nn.Dropout(0.6) # added rn 
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) #  x = self.pool(F.relu(self.conv1(x)))  before 
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        preds = self.fc2(x)
        return preds
    

class CNN_1Test(nn.Module) :

    """
    This class was me testing yet another architecture. 

    A CNN which does classification.

    It should use at least one convolutional layer.
    """
    
    def __init__(self, input_channels, n_classes): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2) 
        
        self.flatten_size = 32*13*13 

        self.fc1 = nn.Linear(self.flatten_size, 128) 
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x): 
        x = F.relu(self.conv1(x)) 
        x = self.pool(x) 
        x = self.dropout(x) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
    
class CNN_3Test(nn.Module): 
    """ 
    and again 
    """

    def __init__(self, input_channels, n_classes): 
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x): 
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        preds = self.fc2(x)
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        import time 
        for ep in range(self.epochs):
            start = time.time()
            self.train_one_epoch(dataloader, ep)
            end = time.time()
            dur = end-start
            print(f'Epoch {ep + 1}/{self.epochs} completed in {dur:.2f} seconds.')

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        running_loss = 0.0 

        for inputs, labels in dataloader: 
            self.optimizer.zero_grad()
            outputs = self.model(inputs) 
            loss = self.criterion(outputs, labels) 
            loss.backward()
            self.optimizer.step()
            running_loss+=loss.item()
            self.optimizer.zero_grad()

        print(f'Epoch {ep + 1}, Loss: {running_loss / len(dataloader)}') 


    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval() 
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                outputs = self.model(inputs)
                _, curr_preds = torch.max(outputs, 1)
                preds.append(curr_preds)

        pred_labels = torch.cat(preds)
        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels).long()) # added .long()
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        return pred_labels.cpu().numpy()