import argparse
# import packages
import os
import random
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix, cohen_kappa_score

from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import *

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# source file for thesis
class InputData():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.input_data = None
        self.X_data = None
        self.y_data = None
        self.y_count = None
        self.minority_class = None
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
        self.scaler = None
    def load_data(self, file_name):
        file_full_dir = os.path.join(self.data_dir, file_name)
        if file_full_dir.endswith('.dat'):
            self.input_data = pd.read_table(file_full_dir, sep='\s\s+', engine='python')
        else:
            self.input_data = pd.read_csv(file_full_dir)
        self.input_data.columns = [self.input_data.columns[c].strip() for c in range(0, len(self.input_data.columns))]
    def X_y_split(self, y_column_name):
        self.X_data = self.input_data.drop([y_column_name], axis = 1)
        self.y_data = self.input_data[y_column_name]
    # convert class to integer ('M'(majority class) -> 0, 'R'(minority class) -> 1)
    def convert_class_to_Integer(self):
        if (self.X_data.dtypes == 'O').any(): # if there is at least one string variable in X
            self.X_data = pd.get_dummies(self.X_data, drop_first = True) # make dummy variables
        self.y_count = self.y_data.value_counts()
        self.minority_class = self.y_count.index[np.argmin(self.y_count)] # find minority class
        self.y_data = [(1 if x == self.minority_class else 0) for x in self.y_data] # change ('M'(majority class) -> 0, 'R'(minority class) -> 1)
    def get_imbalance_ratio(self):
        min_c = self.y_count.values[np.argmin(self.y_count)]
        max_c = self.y_count.values[np.argmax(self.y_count)]
        rho = max_c / min_c
        return rho
    def train_val_test_split(self, valid_size, test_size, random_seed):
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.X_data, self.y_data, test_size = test_size, random_state = random_seed, stratify = self.y_data)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, test_size = valid_size, random_state = random_seed, stratify = y_train)
    def Scaling(self, Scaler):
        self.scaler = Scaler
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_valid = self.scaler.transform(self.X_valid)
        self.X_test = self.scaler.transform(self.X_test)
    def get_data(self):
        return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test
    def sampling(self, sampling_type, sampling_ratio, random_seed, n_neighbors):
        SM = eval(sampling_type + '(sampling_strategy = sampling_ratio, random_state = random_seed, k_neighbors = n_neighbors)')
        self.X_train, self.y_train = SM.fit_resample(self.X_train, self.y_train)
    def get_imbalance_ratio2(self):
        rho = sum([self.y_train[i] == 0 for i in range(0, len(self.y_train))]) / sum([self.y_train[i] == 1 for i in range(0, len(self.y_train))])
        return rho
    
class MyDataset(Dataset):
    # Initialize data
    def __init__(self, X_data, y_data):
        self.len = X_data.shape[0]
        self.x_data = torch.from_numpy(X_data.astype(np.float32))
        self.y_data = torch.tensor(y_data, dtype = torch.float32)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_data, activation_function):
        super(NeuralNetwork, self).__init__()
        self.activation = activation_function
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_data.X_train.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            self.activation
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
class LSTM(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers, seq_length, activation_function): 
        super(LSTM, self).__init__()
        self.num_layers = num_layers #number of layers 
        self.input_size = input_size #input size 
        self.hidden_size = hidden_size #hidden state 
        self.seq_length = seq_length #sequence length 
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #lstm 
        self.fc_1 = nn.Linear(hidden_size, 8) #fully connected 1 
        self.fc = nn.Linear(8, 1) #fully connected last layer 
        self.relu = nn.ReLU() 
        self.activation = activation_function
        
    def forward(self,x): 
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state 
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state

        # Propagate input through LSTM 
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state 
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next 
        out = self.relu(hn) 
        out = self.fc_1(out) #first Dense 
        out = self.relu(out) #relu 
        out = self.fc(out)
        out = self.activation(out) #Final Output return out
        return out

def train(train_dataloader, valid_dataloader, model, loss_fn, optimizer, patience, show_training): # train function for normal case
    train_losses = []
    valid_losses = []
    
    size = len(train_dataloader.dataset)
    
    model.train()
    current = 0
    for batch, (X, y) in enumerate(train_dataloader):

        # Compute prediction error
        pred = torch.reshape(model(X), (-1,))
        loss = loss_fn(pred, y)
        
        # check nan loss
        if np.isnan(loss.detach().numpy()):
            continue

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        train_losses.append(loss)
        if show_training : 
            current += len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    model.eval()
    for batch, (X, y) in enumerate(valid_dataloader):

        # Compute prediction error
        pred = torch.reshape(model(X), (-1,))
        loss = loss_fn(pred, y)
        
        # check nan loss
        if np.isnan(loss.detach().numpy()):
            continue

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        valid_losses.append(loss)
        
    if np.isnan(np.average(train_losses)):
        print()
    
    return np.average(train_losses), np.average(valid_losses), None, None, None

def train2(train_dataloader, valid_dataloader, model, loss_fn, optimizer, patience, show_training): # train function for tracking mu, sigma, xi
    train_losses = []
    valid_losses = []
    mu = []
    sigma = []
    xi = []
    
    size = len(train_dataloader.dataset)
    
    model.train()
    current = 0
    for batch, (X, y) in enumerate(train_dataloader):

        # Compute prediction error
        pred = torch.reshape(model(X), (-1,))
        loss = loss_fn(pred, y)
        
        # check nan loss
        if np.isnan(loss.detach().numpy()):
            continue

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        train_losses.append(loss)
        
        mu.append(model.activation.mu.item())
        sigma.append(model.activation.sigma.item())
        xi.append(model.activation.xi.item())
        
        if show_training : 
            current += len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    model.eval()
    for batch, (X, y) in enumerate(valid_dataloader):

        # Compute prediction error
        pred = torch.reshape(model(X), (-1,))
        loss = loss_fn(pred, y)
        
        # check nan loss
        if np.isnan(loss.detach().numpy()):
            continue

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        valid_losses.append(loss)
        
    if np.isnan(np.average(train_losses)):
        print()
    
    return np.average(train_losses), np.average(valid_losses), np.average(mu), np.average(sigma), np.average(xi)

def test(dataloader, model, loss_fn,  show_training):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = torch.reshape(model(X), (-1,))
            test_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    if show_training:
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

# Type 1: For xi = 0 (Gumbel)
def t1(x, mu, sigma):
    return torch.exp(-torch.exp(- (x - mu) / sigma))

# Type 2: For xi > 0 (Frechet) or xi < 0 (Reversed Weibull)
def t23(x, mu, sigma, xi):
    y = (x - mu) / sigma
    y = xi * y
    y = torch.max(-torch.tensor(1.), y)
    y = torch.exp(-torch.pow(torch.tensor(1.) + y, -torch.tensor(1.) / xi))
    return y
    
# Inherit from Function
class GEVFunction(torch.autograd.Function):
    
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, mu, sigma, xi):
        ctx.save_for_backward(input, mu, sigma, xi)
        sigma = torch.max(torch.tensor(1e-10), sigma) # sigma < 0 doesn't make sense
        output = torch.where(torch.tensor(torch.equal(torch.tensor(0.), xi)), t1(input, mu, sigma), t23(input, mu, sigma, xi)) # This chooses the type based on xi
        return output
    
    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, mu, sigma, xi = ctx.saved_tensors
        grad_input = grad_mu = grad_sigma = grad_xi = None
        if torch.equal(torch.tensor(0.), xi): #Type 1: xi = 0 (Gumbel)
            common_term = (1 / sigma) * torch.exp(- (input - mu) / sigma) * torch.exp(-torch.exp(- (input - mu) / sigma))
            grad_input = grad_output * common_term
            grad_mu = grad_output * -common_term
            grad_sigma = grad_output * - (input - mu) / sigma
            grad_xi = torch.tensor(0.)
            
        else:  # Type 2: For xi > 0 (Frechet) or xi < 0 (Reversed Weibull)
            batch_size = 32
            # change the type of tensors
            input = input.flatten()
            input = tf.Variable(input.tolist(), name = 'input')
            mu = tf.Variable(mu.tolist() * batch_size, name = 'mu')
            sigma = tf.Variable(sigma.tolist() * batch_size, name = 'sigma')
            xi = tf.Variable(xi.tolist() * batch_size, name = 'xi')
            
            # get grads using numerical diff (multi-processing)
            with tf.GradientTape() as tape:
                Q = (input - mu) / sigma
                Q = xi * Q
                Q = tf.math.maximum(tf.constant([-1.]), Q)
                Q = tf.math.exp(-tf.math.pow(tf.constant([1.]) + Q + 1e-7, -tf.constant([1.]) / xi)) # add epsilon to prevent 0^j.

            grads = tape.gradient(Q, [input, mu, sigma, xi])
            input_grad = torch.tensor(grads[0].numpy())
            mu_grad = torch.tensor(grads[1].numpy())
            sigma_grad = torch.tensor(grads[2].numpy())
            xi_grad = torch.tensor(grads[3].numpy())
            
            grad_input = grad_output * input_grad
            grad_mu = grad_output * mu_grad
            grad_sigma = grad_output * sigma_grad
            grad_xi = grad_output * xi_grad
            
        return grad_input, grad_mu, grad_sigma, grad_xi
    
# Inherit from Module
class GEV(nn.Module):
    def __init__(self, input_features, output_features):
        super(GEV, self).__init__()
        
        self.mu = torch.nn.parameter.Parameter(torch.tensor(0.))
        self.sigma = torch.nn.parameter.Parameter(torch.tensor(1.))
        self.xi = torch.nn.parameter.Parameter(torch.tensor(0.01))
        self.eps = torch.tensor(1e-10)
        self.zero = torch.tensor(0.)
        self.one = torch.tensor(1.)
        
    def forward(self, input):
        return GEVFunction.apply(input, self.mu, self.sigma, self.xi)
    
    def extra_repr(self):
        return 'mu={}, sigma={}, xi={}'.format(
            self.mu, self.sigma, self.xi is not None
        )

def assess_model(model, X_test, y_test, threshold, pos_label) :
    if threshold is None:
        # prediction for test data
        model.eval()
        with torch.no_grad():
            test_pred = torch.reshape(model(torch.from_numpy(X_test.astype(np.float32))), (-1, ))
        fpr, tpr, thresholds = roc_curve(y_test, test_pred, pos_label = pos_label)
        pred_test = test_pred > 0.5

        # get Assessment Indexes
        AUC = auc(fpr,tpr)
        F1 = f1_score(y_true = y_test, y_pred = pred_test, pos_label = pos_label)
        Kappa = cohen_kappa_score(y_test, pred_test)
        confusion__matrix = confusion_matrix(y_true = y_test, y_pred = pred_test)
        FP = confusion__matrix[0,1]
        FN = confusion__matrix[1,0]
        TP = confusion__matrix[1,1]
        TN = confusion__matrix[0,0]

        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)
        tnr = 1 - fpr

        GM = np.sqrt(tpr * tnr)
        BA = ((tpr + tnr) / 2)
        BIA = sum([(y_test[i] - test_pred.tolist()[i]) ** 2 for i in range(0, len(y_test))]) * 2 / len(y_test)

    else :
        # prediction for test data
        model.eval()
        with torch.no_grad():
            test_pred = torch.reshape(model(torch.from_numpy(X_test.astype(np.float32))), (-1, ))
        fpr, tpr, thresholds = roc_curve(y_test, test_pred, pos_label = pos_label)
        tnr = 1 - fpr
        GM = np.sqrt(tpr * tnr)
        pred_test = test_pred > threshold

        # get Assessment Indexes
        AUC = auc(fpr,tpr)
        F1 = f1_score(y_true = y_test, y_pred = pred_test, pos_label = pos_label)
        Kappa = cohen_kappa_score(y_test, pred_test)
        BA = ((tpr + tnr) / 2)[np.argmax(GM)]
        GM = GM[np.argmax(GM)]
        BIA = sum([(y_test[i] - test_pred.tolist()[i]) ** 2 for i in range(0, len(y_test))]) * 2 / len(y_test)

    return fpr, tpr, AUC, F1, BA, GM, BIA, Kappa

def get_proper_threshold(model, X_valid, y_valid, pos_label) :
    
    # prediction for test data
    model.eval()
    with torch.no_grad():
        valid_pred = torch.reshape(model(torch.from_numpy(X_valid.astype(np.float32))), (-1, ))
    fpr, tpr, thresholds = roc_curve(y_valid, valid_pred, pos_label = pos_label)
    tnr = 1 - fpr
    GM = np.sqrt(tpr * tnr)
    return thresholds[np.argmax(GM)]

def save_ROC(plot_path, model, X_test, y_test, pos_label, AUC):
    # prediction for test data
    model.eval()
    with torch.no_grad():
        test_pred = torch.reshape(model(torch.from_numpy(X_test.astype(np.float32))), (-1, ))
    fpr, tpr, thresholds = roc_curve(y_test, test_pred, pos_label = pos_label)
    
    # visualize ROC curve
    fig = plt.figure(figsize=(10,8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(plot_path)
    plt.close(fig)

def save_training_plot(plot_path, avg_train_losses, avg_valid_losses):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label = 'Training Loss')
    plt.plot(range(1, len(avg_valid_losses) + 1), avg_valid_losses, label = 'Validation Loss')

    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses)) + 1 
    plt.axvline(minposs, linestyle = '--', color = 'r', label = 'Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, np.max(avg_train_losses + avg_valid_losses)) # consistent scale
    plt.xlim(0, len(avg_train_losses) + 1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    
def save_parameters_plot(plot_path, mu, sigma, xi):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(mu) + 1), mu, label = 'Mu')
    plt.plot(range(1, len(sigma) + 1), sigma, label = 'Sigma')
    plt.plot(range(1, len(xi) + 1), xi, label = 'Xi')

    plt.xlabel('epochs')
    plt.ylabel('parameters')
    plt.ylim(np.min(mu + sigma + xi) - 0.1, np.max(mu + sigma + xi) + 0.1) # consistent scale
    plt.xlim(0, len(xi) + 1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    
def pass_counter(message):
    pass

class MSFELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MSFELoss, self).__init__()

    def forward(self, inputs, targets):
        # FPE
        neg_class_cond = targets.type(torch.float) == torch.tensor(0, dtype = torch.float)
        N = neg_class_cond.sum()
        FPE = torch.where(neg_class_cond, inputs.type(torch.float)**2, torch.tensor(0, dtype = torch.float)).sum() / N
        
        # FNE
        pos_class_cond = targets.type(torch.float) == torch.tensor(1, dtype = torch.float)
        P = pos_class_cond.sum()
        FNE = torch.where(pos_class_cond, (1-inputs).type(torch.float)**2, torch.tensor(0, dtype = torch.float)).sum() / P
        
        MSFE = FPE**2 + FNE**2
        return MSFE
    
class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2.0, epsilon = 1e-7, weight=None, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = torch.tensor(epsilon)

    def forward(self, inputs, targets):
        # p_t
        p_t = torch.where(targets.type(torch.float) == torch.tensor(1, dtype = torch.float), inputs.type(torch.float), 1 - inputs.type(torch.float))
        
        modulating_factor = (1 - p_t) ** self.gamma
        
        p_t = torch.where(abs(p_t) < self.eps, p_t + self.eps, p_t) # prevent log(0) in loss function
            
        FL = - self.alpha * modulating_factor * torch.log(p_t)
        
        return torch.sum(FL)

'''
    code refactoring
'''
def training(epochs, train_dataloader, valid_dataloader, model, loss_fn, optimizer, patience, show_training, model_path):

    avg_train_losses = []
    avg_valid_losses = []
    
    avg_mu = []
    avg_sigma = []
    avg_xi = []

    # initialize the early_stopping object
    if show_training :
        counter = print
    else :
        counter = pass_counter
    early_stopping = EarlyStopping(patience = patience, verbose = show_training, path = model_path, trace_func = counter)

    for t in range(epochs):
        
        if show_training :
            print(f"Epoch {t+1}\n-------------------------------")
        
        if os.path.splitext(model_path)[0].endswith('GEV'): # if gev model
            train_loss, valid_loss, mu, sigma, xi = train2(train_dataloader, valid_dataloader, model, loss_fn, optimizer, patience, show_training)
        else:
            train_loss, valid_loss, mu, sigma, xi = train(train_dataloader, valid_dataloader, model, loss_fn, optimizer, patience, show_training)
        
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        if mu is None:
            avg_mu, avg_sigma, avg_xi = None, None, None
        else:
            avg_mu.append(mu)
            avg_sigma.append(sigma)
            avg_xi.append(xi)
        
        #test(test_dataloader, model, loss_fn, show_training)

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    return avg_train_losses, avg_valid_losses, model, avg_mu, avg_sigma, avg_xi

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def main(working_directory_list, data_source_list, oversampling_list, randomseed_list, test_size_list, valid_size_lilst, learning_rate_list, batch_size_list, epochs_list, patience_list, show_training_list, loss_fn_list, activation_list):
    
    if(len(working_directory_list) == 0):
        print('Please give working directory')
    if(len(data_source_list) == 0):
        print('Please give directory that has *.csv files')
    
    working_directory = working_directory_list[0]
    data_source = data_source_list[0]
    oversampling = oversampling_list[0]
    random_seed = randomseed_list
    test_size = test_size_list[0]
    valid_size = valid_size_lilst[0]
    learning_rate = learning_rate_list[0]
    batch_size = batch_size_list[0]
    epochs = epochs_list[0]
    patience = patience_list[0]
    show_training = show_training_list[0]
    loss_fn = loss_fn_list[0]
    activation = activation_list[0]

    # Casting
    oversampling = float(oversampling)
    test_size = float(test_size)
    valid_size = float(valid_size)
    learning_rate = float(learning_rate)
    batch_size = int(batch_size)
    epochs = int(epochs)
    patience = int(patience)
    show_training = bool(show_training)
    random_seed = [int(random_seed[i]) for i in range(0, len(random_seed))]

    # print settings
    print('\n================ Directory Settings ================\n')
    print('Working Directory : {}'.format(working_directory))
    print('Data Source : {}'.format(data_source))
    print('\n================ Data Settings ================\n')
    print('Train Data Set Size : {} %'.format( 100 - round(valid_size * (1 - test_size) * 100, 1) - round( test_size * 100, 1)))
    print('Validation Data Set Size : {} %'.format( round(valid_size * (1 - test_size) * 100, 1)))
    print('Test Data Set Size : {} %'.format( round( test_size * 100, 1) ) )
    print('Over Sampling Ratio : {} => {}'.format(oversampling, str(round(1/oversampling, 1)) + '(Major)' + ' : ' + str(1) + '(Minor)'))
    print('\n================ Training Settings ================\n')
    print('Activation Function : {}'.format(activation))
    print('Loss Function : {}'.format(loss_fn))
    print('Learning rate : {}'.format(learning_rate))
    print('Batch Size : {}'.format(batch_size))
    print('Epochs : {}'.format(epochs))
    print('Early Stopping patience : {}'.format(patience))
    print('\n================ Experiment Settings ================\n')
    print('Random Seed : {}'.format(random_seed))
    print('Show Training Procedure : {}'.format(show_training))
    print('\n')

    if loss_fn == 'Focal':
        loss_fn = FocalLoss()
    elif loss_fn == 'BCE':
        loss_fn = nn.BCELoss()
    else:
        print('\n {} : Unsupported Loss function'.format(loss_fn))

    print('\n\n================ Start Main ================  ' +  str(datetime.datetime.now()))

    if activation == 'Sigmoid':
        main_sigmoid(working_directory, data_source, oversampling, test_size, valid_size, learning_rate, batch_size, epochs, patience, show_training, loss_fn, activation, random_seed)
    elif activation == 'GEV':
        main_gev(working_directory, data_source, oversampling, test_size, valid_size, learning_rate, batch_size, epochs, patience, show_training, loss_fn, activation, random_seed)
    else:
        print('\n {} : Unsupported activation function'.format(activation))
        quit()
    
def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(nargs='+', help = 'Example) F:/imb_class', dest = 'working_directory')
    parser.add_argument('--source', '-s', nargs = '*', help = 'Example) F:/imb_class/data', default=[], dest = 'data_source')
    parser.add_argument('--oversampling', '-os', nargs = '*', help = 'Example) 0.1', default = [1/20], dest = 'oversampling')
    parser.add_argument('--seed', '-r', nargs = '*', help = 'Example) 20210905 20210906 20210907', default = list(range(20210905, 20210908)), dest = 'randomseed')
    parser.add_argument('--testsize', '-t', nargs = '*', help = 'Example) 0.3', default = [0.3], dest = 'test_size')
    parser.add_argument('--validsize', '-v', nargs = '*', help = 'Example) 0.2', default = [0.2], dest = 'valid_size')
    parser.add_argument('--learningrate', '-l', nargs = '*', help = 'Example) 0.001', default = [0.001], dest = 'learning_rate')
    parser.add_argument('--batchsize', '-b', nargs = '*', help = 'Example) 32', default = [32], dest = 'batch_size')
    parser.add_argument('--epochs', '-e', nargs = '*', help = 'Example) 2000', default = [2000], dest = 'epochs')
    parser.add_argument('--patience', '-p', nargs = '*', help = 'Example) 20', default = [20], dest = 'patience')
    parser.add_argument('--showtraining', '-show', nargs = '*', help = 'Example) True or False', default = [False], dest = 'show_training')
    parser.add_argument('--lossfunction', '-loss', nargs = '*', help = 'Example) Focal Loss or Binary Cross Entropy', default = ['Focal'], dest = 'loss_fn', choices=['Focal', 'BCE'])
    parser.add_argument('--activation', '-a', nargs = '*', help = 'Example) Sigmoid or GEV', default = ['GEV'], dest = 'activation', choices = ['Sigmoid', 'GEV'])
    
    working_directory_list = parser.parse_args().working_directory
    data_source_list = parser.parse_args().data_source
    oversampling_list = parser.parse_args().oversampling
    randomseed_list = parser.parse_args().randomseed
    test_size_list = parser.parse_args().test_size
    valid_size_list = parser.parse_args().valid_size
    learning_rate_list = parser.parse_args().learning_rate
    batch_size_list = parser.parse_args().batch_size
    epochs_list = parser.parse_args().epochs
    patience_list = parser.parse_args().patience
    show_training_list = parser.parse_args().show_training
    loss_fn_list = parser.parse_args().loss_fn
    activation_list = parser.parse_args().activation
    
    
    return working_directory_list, data_source_list, oversampling_list, randomseed_list, test_size_list, valid_size_list, learning_rate_list, batch_size_list, epochs_list, patience_list, show_training_list, loss_fn_list, activation_list

def main_gev(working_directory, data_source, oversampling, test_size, valid_size, learning_rate, batch_size, epochs, patience, show_training, loss_fn, activation, random_seed):
    files = [_ for _ in os.listdir(data_source) if _.endswith('.csv')]
    for f in range(0, len(files)):
        results = list()
        print('\n' + files[f] + '\n')
        
        # make directory
        last_directory_name = os.path.join(os.path.split(data_source)[0], activation)
        result_dir = os.path.join(working_directory, 'result', last_directory_name)
        if not os.path.exists(result_dir) :
            os.makedirs(result_dir)
        # make directory for training plot
        result_train_plot_dir = os.path.join(working_directory, 'result',  last_directory_name, 'training_plot')
        if not os.path.exists(result_train_plot_dir) :
            os.makedirs(result_train_plot_dir)
        # make directory for roc plot
        result_roc_plot_dir = os.path.join(working_directory, 'result',  last_directory_name, 'roc_plot')
        if not os.path.exists(result_roc_plot_dir) :
            os.makedirs(result_roc_plot_dir)
        # make directory for parameters plot
        result_param_plot_dir = os.path.join(working_directory, 'result',  last_directory_name, 'param_plot')
        if not os.path.exists(result_param_plot_dir) :
            os.makedirs(result_param_plot_dir)
        # make directory for model
        result_model_dir = os.path.join(working_directory, 'result',  last_directory_name, 'trained_model')
        if not os.path.exists(result_model_dir) :
            os.makedirs(result_model_dir)
        fin_result_path = os.path.join(result_dir, os.path.splitext(files[f])[0] + '_result.csv')
        
        if os.path.isfile(fin_result_path):
            print('Already exists')
            continue
            
        else:
        
            # read Data
            input_data = InputData(data_source)
            input_data.load_data(file_name = files[f])
            input_data.X_y_split(y_column_name = 'y')
            input_data.convert_class_to_Integer()
            rho = input_data.get_imbalance_ratio()
            
            result_os = []
                    
            # iterate by seed
            for s in range(0, len(random_seed)):
                rho2 = rho

                # set seed
                seed = random_seed[s]
                seed_everything(seed)

                # preprocess data
                input_data.train_val_test_split(valid_size = valid_size, test_size = test_size, random_seed = seed)
                if rho > (1 / oversampling) :
                    
                    # oversampling using SMOTE
                    for n_neighbors in range(5, 0, -1):
                        try:
                            input_data.sampling(sampling_type = 'SMOTE', sampling_ratio = oversampling, random_seed = seed, n_neighbors = n_neighbors)
                            rho2 = input_data.get_imbalance_ratio2()
                            break
                        except:
                            pass

                input_data.Scaling(Scaler = MinMaxScaler(feature_range = (0, 1)))
                X_train, y_train, X_valid, y_valid, X_test, y_test = input_data.get_data()

                # Create data set
                training_data = MyDataset(X_train, y_train)
                valid_data = MyDataset(X_valid, y_valid)
                test_data = MyDataset(X_test, y_test)

                # Create data loaders
                train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle = True)
                valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
                test_dataloader = DataLoader(test_data, batch_size=batch_size)

                # Create models
                model_gev = NeuralNetwork(input_data = input_data, activation_function = GEV(8, 1))

                # Define optimizer
                optimizer_gev = torch.optim.Adam(model_gev.parameters(), lr = learning_rate)

                # train models
                gev_model_path = os.path.join(result_model_dir, os.path.splitext(files[f])[0] + '_' + str(seed) + '_' + activation + '.pt')

                print('training GEV Neural Net ...')
                print(datetime.datetime.now())
                avg_train_losses_gev, avg_valid_losses_gev, model_gev, mu, sigma, xi = training(epochs, train_dataloader, valid_dataloader, model_gev, loss_fn, optimizer_gev, patience, show_training, gev_model_path)

                # load the last checkpoint with the best model
                model_gev.load_state_dict(torch.load(gev_model_path))

                # Calculate Test loss with the best model
                avg_test_loss_gev = test(test_dataloader, model_gev, loss_fn, show_training)

                # get best threshold using Geometric-Mean
                threshold_gev = get_proper_threshold(model_gev, input_data.X_valid, input_data.y_valid, pos_label = 1)

                # Assess models using evaluation index
                fpr_gev, tpr_gev, AUC_gev, F1_gev, BA_gev, GM_gev, BIA_gev, Kappa_gev = assess_model(model_gev, input_data.X_test, input_data.y_test, threshold = threshold_gev, pos_label = 1)
                AUC = [AUC_gev]; F1 = [F1_gev]; BA = [BA_gev]; GM = [GM_gev]; BIA = [BIA_gev]; Kappa = [Kappa_gev] # concat indexes

                # make dataframe
                result_dict = {'seed' : [seed] , 'dataset' : [os.path.splitext(files[f])[0]] , 'Imbalance-ratio' : [rho2], 'activation' : [activation], 
                            "F1-score" : F1, 'Geometric-Mean' : GM, 'Area Under Curve' : AUC, 'Balanced_Accuracy' : BA, 'Brier Inaccuracy' : BIA, "Cohen Kappa" : Kappa, 'Test_loss' : [avg_test_loss_gev], 
                            'Mu' : mu[len(mu)-1], 'Sigma' : sigma[len(sigma)-1], 'Xi' : xi[len(xi)-1]}
                results.append(pd.DataFrame(result_dict))
                # exec('result' + '_' + str(s) + '= pd.DataFrame(result_dict)')

                # save ROC Curve
                plot_path_gev = os.path.join(result_roc_plot_dir, 'roc_' + activation + '_' + os.path.splitext(files[f])[0] + '_' + str(s) + '.png')
                save_ROC(plot_path_gev, model_gev, input_data.X_test, input_data.y_test, pos_label = 1, AUC = AUC_gev)

                # save training plot
                plot_path_gev = os.path.join(result_train_plot_dir, 'train_' + activation + '_' + os.path.splitext(files[f])[0] + '_' + str(s) + '.png')
                save_training_plot(plot_path_gev, avg_train_losses_gev, avg_valid_losses_gev)

                # save parameters plot
                if mu is not None:
                    plot_path_gev = os.path.join(result_param_plot_dir, '_' + activation + '_parameters_' + os.path.splitext(files[f])[0] + '_' + str(s) + '.png')
                    save_parameters_plot(plot_path_gev, mu, sigma, xi)

                print(f"        randomseed : {seed}, finished!")
                print("        " + str(datetime.datetime.now()))

            # close all open figures
            plt.close('all')
            # concat result
            result = pd.concat(results, axis = 0)
            #result = pd.concat([eval('result_' + str(s)) for s in range(0, len(random_seed))], axis = 0)
            result_os.append(result)
        result = pd.concat(result_os, axis = 0)
        # save result
        result.to_csv(os.path.join(result_dir, os.path.splitext(files[f])[0] + '_result.csv'), index = False)
    
def main_sigmoid(working_directory, data_source, oversampling, test_size, valid_size, learning_rate, batch_size, epochs, patience, show_training, loss_fn, activation, random_seed):
    files = [_ for _ in os.listdir(data_source) if _.endswith('.csv')]
    for f in range(0, len(files)):
        results = list()
        print('\n' + files[f] + '\n')
    
        # make directory
        last_directory_name = os.path.join(os.path.split(data_source)[0], activation)
        result_dir = os.path.join(working_directory, 'result', last_directory_name)
        if not os.path.exists(result_dir) :
            os.makedirs(result_dir)
        # make directory for training plot
        result_train_plot_dir = os.path.join(working_directory, 'result',  last_directory_name, 'training_plot')
        if not os.path.exists(result_train_plot_dir) :
            os.makedirs(result_train_plot_dir)
        # make directory for roc plot
        result_roc_plot_dir = os.path.join(working_directory, 'result',  last_directory_name, 'roc_plot')
        if not os.path.exists(result_roc_plot_dir) :
            os.makedirs(result_roc_plot_dir)
        # make directory for model
        result_model_dir = os.path.join(working_directory, 'result',  last_directory_name, 'trained_model')
        if not os.path.exists(result_model_dir) :
            os.makedirs(result_model_dir)
        fin_result_path = os.path.join(result_dir, os.path.splitext(files[f])[0] + '_result.csv')
        
        if os.path.isfile(fin_result_path):
            print('Already exists')
            continue
            
        else:


            # read Data
            input_data = InputData(data_source)
            input_data.load_data(file_name = files[f])
            input_data.X_y_split(y_column_name = 'y')
            input_data.convert_class_to_Integer()
            rho = input_data.get_imbalance_ratio()

            # iterate by seed
            for s in range(0, len(random_seed)):
                # set seed
                seed = random_seed[s]
                seed_everything(seed)

                # preprocess data
                input_data.train_val_test_split(valid_size = valid_size, test_size = test_size, random_seed = seed)
                input_data.Scaling(Scaler = MinMaxScaler(feature_range = (0, 1)))
                X_train, y_train, X_valid, y_valid, X_test, y_test = input_data.get_data()

                # preprocess data
                input_data.train_val_test_split(valid_size = valid_size, test_size = test_size, random_seed = seed)
                if rho > (1 / oversampling) :
                                
                    # oversampling using SMOTE
                    for n_neighbors in range(5, 0, -1):
                        try:
                            input_data.sampling(sampling_type = 'SMOTE', sampling_ratio = oversampling, random_seed = seed, n_neighbors = n_neighbors)
                            rho2 = input_data.get_imbalance_ratio2()
                            break
                        except:
                            pass
                input_data.Scaling(Scaler = MinMaxScaler(feature_range = (0, 1)))
                X_train, y_train, X_valid, y_valid, X_test, y_test = input_data.get_data()

                # Create data set
                training_data = MyDataset(X_train, y_train)
                valid_data = MyDataset(X_valid, y_valid)
                test_data = MyDataset(X_test, y_test)

                # Create data loaders
                train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle = True)
                valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
                test_dataloader = DataLoader(test_data, batch_size=batch_size)

                # Create models
                model_sigmoid = NeuralNetwork(input_data = input_data, activation_function = nn.Sigmoid())

                # Define optimizer
                optimizer_sigmoid = torch.optim.Adam(model_sigmoid.parameters(), lr = learning_rate)

                # train models
                sigmoid_model_path = os.path.join(result_model_dir, os.path.splitext(files[f])[0] + '_' + str(seed) + '_sigmoid.pt')

                print('training sigmoid Neural Net ...')
                print(datetime.datetime.now())
                avg_train_losses_sigmoid, avg_valid_losses_sigmoid, model_sigmoid, mu, sigma, xi = training(epochs, train_dataloader, valid_dataloader, model_sigmoid, loss_fn, optimizer_sigmoid, patience, show_training, sigmoid_model_path)

                # load the last checkpoint with the best model
                model_sigmoid.load_state_dict(torch.load(sigmoid_model_path))

                # Calculate Test loss with the best model
                avg_test_loss_sigmoid = test(test_dataloader, model_sigmoid, loss_fn, show_training)

                # Assess models using evaluation index
                fpr_sigmoid, tpr_sigmoid, AUC_sigmoid, F1_sigmoid, BA_sigmoid, GM_sigmoid, BIA_sigmoid, Kappa_sigmoid = assess_model(model_sigmoid, input_data.X_test, input_data.y_test, threshold = None, pos_label = 1)
                AUC = [AUC_sigmoid]; F1 = [F1_sigmoid]; BA = [BA_sigmoid]; GM = [GM_sigmoid]; BIA = [BIA_sigmoid]; Kappa = [Kappa_sigmoid] # concat indexes

                # make dataframe
                result_dict = {'seed' : [seed] , 'dataset' : [os.path.splitext(files[f])[0]] , 'Imbalance-ratio' : [rho] , 'activation' : ['sigmoid'], 
                            "F1-score" : F1, 'Geometric-Mean' : GM, 'Area Under Curve' : AUC, 'Balanced_Accuracy' : BA, 'Brier Inaccuracy' : BIA, "Cohen Kappa" : Kappa, 'Test_loss' : [avg_test_loss_sigmoid],
                            'Mu' : np.nan, 'Sigma' : np.nan, 'Xi' : np.nan}
                results.append(pd.DataFrame(result_dict))

                # save ROC Curve
                plot_path_sigmoid = os.path.join(result_roc_plot_dir, 'roc_sigmoid_' + os.path.splitext(files[f])[0] + '_' + str(s) + '.png')
                save_ROC(plot_path_sigmoid, model_sigmoid, input_data.X_test, input_data.y_test, pos_label = 1, AUC = AUC_sigmoid)

                # save training plot
                plot_path_sigmoid = os.path.join(result_train_plot_dir, 'train_sigmoid_' + os.path.splitext(files[f])[0] + '_' + str(s) + '.png')
                save_training_plot(plot_path_sigmoid, avg_train_losses_sigmoid, avg_valid_losses_sigmoid)

                print(f"        randomseed : {seed}, finished!")
                print("        " + str(datetime.datetime.now()))

            # close all open figures
            plt.close('all')
            # concat result
            result = pd.concat(results, axis = 0)
            # result = pd.concat([eval('result_' + str(s)) for s in range(0, len(random_seed))], axis = 0)
            # save result
            result.to_csv(fin_result_path, index = False)

if __name__ == '__main__':
    working_directory_list, data_source_list, oversampling_list, randomseed_list, test_size_list, valid_size_list, learning_rate_list, batch_size_list, epochs_list, patience_list, show_training_list, loss_fn_list, activation_list = get_arguments()
    main(working_directory_list, data_source_list, oversampling_list, randomseed_list, test_size_list, valid_size_list, learning_rate_list, batch_size_list, epochs_list, patience_list, show_training_list, loss_fn_list, activation_list)