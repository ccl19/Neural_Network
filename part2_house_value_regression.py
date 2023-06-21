import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.base import BaseEstimator
from datetime import datetime

# import printdata

np.set_printoptions(precision=5, threshold=30, edgeitems=7)
np.core.arrayprint._line_width = 120
np.random.seed(3407)
torch.manual_seed(3407)
# select loss function https://pytorch.org/docs/stable/nn.html#loss-functions
#select optimiser, stohastic gradient decent (SGD); lr = learning rate https://pytorch.org/docs/stable/optim.html#algorithms


class Regressor(nn.Module):

    def __init__(
        self, 
        x, 
        nb_epoch = 99999, #usually bypassed, due to patience
        use_patience = True,
        val_prop = 0.11, #by default it is 9.9% of entire dataset 
        nb_neurons_per_hidden_layer = [32,32,16,8,4],
        minibatch_size = 128, #only go in power of 2
        lr = 0.005,
        patience = 200,
        activation = nn.ReLU, #fastest learning
        optimizer_name = torch.optim.Adam,
        criterion = torch.nn.MSELoss
        ):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        print("\n NEW NETWORK")

        super(Regressor, self).__init__() #extend nn.Module
        self.x = x  #for sklearn
        Y = pd.DataFrame()
        X, Y = self._preprocessor(x, Y, training = True)      # Pytorch returns a tuple which is composed of the input[X] and output arrays[Y]
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.use_patience = use_patience
        self.val_prop = val_prop
        self.minibatch_size = min(minibatch_size, len(X))
        self.nb_neurons_per_hidden_layer = nb_neurons_per_hidden_layer
        self.nb_neurons_per_layer = [self.input_size] + self.nb_neurons_per_hidden_layer + [self.output_size]
        self.nb_linears = len(self.nb_neurons_per_layer) - 1      # Each layer has linear transformation except output

        self.linears = nn.ModuleList([])   #stores all linear layers
        
        print("nb_epoch", self.nb_epoch)
        print("nb_neurons_per_layer: ", self.nb_neurons_per_layer)
        for i in range(self.nb_linears):
            self.linears.append(nn.Linear(self.nb_neurons_per_layer[i], self.nb_neurons_per_layer[i+1]))

        self.activation = activation()

        self.lr = lr
        self.optimizer_name = optimizer_name
        self.criterion = criterion()
        
        self.min_loss = np.inf
        self.converge_count = 0
        self.patience = patience
        
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################

        if isinstance(x, pd.Series):
            x = x.to_frame().T

        if isinstance(x, pd.DataFrame): #idempotency
            if training:
                self.binarizer = LabelBinarizer()
                self.binarizer.fit(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])
                ocean_proximity_df = pd.DataFrame(self.binarizer.transform(x["ocean_proximity"]),
                                        columns=self.binarizer.classes_, index=x.index)  # one-hot encoding for the 5 categorical values
                binarised_x = x.loc[:, x.columns != "ocean_proximity"].join(ocean_proximity_df)
                binarised_x.reset_index(inplace=True, drop=True)
                #numpy array:
                # median imputation may be problematic because this is multivariate analysis, however still least biased option
                self.imputer = SimpleImputer(strategy="median") # median is used because data is skewed (not normal distribution)
                imputed_x = self.imputer.fit_transform(binarised_x) # replace nan with median value of column (and convert to numpy array)
                self.scaler = MinMaxScaler() # initialise new scaler
                norm_x = self.scaler.fit_transform(imputed_x) # learn feature-wise min & max and also normalise
            else: # if testing
                ocean_proximity_df = pd.DataFrame(self.binarizer.transform(x["ocean_proximity"]),
                                                  columns=self.binarizer.classes_, index=x.index)  # one-hot encoding for the 5 categorical values
                binarised_x = x.loc[:, x.columns != "ocean_proximity"].join(ocean_proximity_df)
                binarised_x.reset_index(inplace=True, drop=True)
                #numpy array:
                imputed_x = self.imputer.transform(binarised_x) #Median of train and test set should be equal. This avoids issues with small test sets
                norm_x = self.scaler.transform(imputed_x) # normalise using training set's min & max
            # target is not normalised as it doesn't affect descent stepsize

            output_x = torch.tensor(norm_x).float()
        else:
            output_x = x #already preprocessed

        if isinstance(y, pd.Series):
            y = y.to_frame().T

        if y is None:
            output_y = None
        elif isinstance(y, pd.DataFrame):
            if not y.empty:
                y.reset_index(inplace=True, drop=True)
                output_y = torch.tensor(np.array(y)).float()
            else:
                output_y = y    #empty Y from init
        else:
            output_y = y    #most likely tensor
        
        # Return preprocessed x and y, return None for y if it was None
        return output_x, output_y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1). EXPECTED Y

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        print("\n TRAINING")

        self.optimizer = self.optimizer_name(self.parameters(), lr = self.lr) #has to be here for sklearn cloning

        X, Y = self._preprocessor(x, y, training = True) # Do not forget

        print("train+val shape: ", X.shape) #####

        val_size = int(len(X) * self.val_prop) #splitting train and val here to conform to LabTS format (as we are using patience)
        X_val = X[: val_size]
        Y_val = Y[: val_size]
        X_train = X[val_size :]
        Y_train = Y[val_size :]

        print("X train shape: ", X_train.shape) #####
        print("Y train shape: ", Y_train.shape) #####
        print("X val shape: ", X_val.shape) #####

        X_minibatches = torch.split(X_train, self.minibatch_size, dim = 0)
        Y_minibatches = torch.split(Y_train, self.minibatch_size, dim = 0)
        
        print("X_minibatches length: ", len(X_minibatches)) #####
        print("Y_minibatches length: ", len(Y_minibatches)) #####

        # optimizer = torch.optim.Adam([self.model.parameters()], lr = 0.0001)                       #parameters can only be Tensors
        for epoch in range(self.nb_epoch): #loop through epochs (will usually stop before, using patience)
            for minibatchX, minibatchY in zip(X_minibatches,Y_minibatches): #loop through minibatches
                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                
                # Forward pass: Compute predicted y by passing x to the model
                # pred_y = Regressor.predict(X) # as the predict is already called in score
            
                # Compute and print loss
                outputs = self.predict(minibatchX, numpy=False)
                loss = self.criterion(outputs, minibatchY)
                loss.backward()
                    
                self.optimizer.step() #update weights and biases

            # Patience for every epoch
            epoch_loss = self.score(X_val, Y_val)   #val set used for early stopping to prevent overfitting

            if epoch_loss < self.min_loss:
                self.min_loss = epoch_loss
                self.converge_count = 0 # reset patience counter 
            else:
                self.converge_count += 1 # evidence for convergence (not improving)

            if not epoch % 100: ###
                root_epoch_loss = np.sqrt(epoch_loss)
                print(f'epoch: {epoch}\t MSE: {epoch_loss:.1f} \t RMSE: {root_epoch_loss:.3f} \t {self.nb_neurons_per_hidden_layer} \t {self.minibatch_size} \t {self.patience} \t {self.lr} \t {self.optimizer_name}')
                if epoch >= 100 and root_epoch_loss > 200000:
                    print(f"Terminated at epoch {epoch} due to degeneracy")
                    return self #terminate loop

            if self.use_patience and self.converge_count >= self.patience:
                print(f"Terminated at epoch {epoch+1} using patience") # Prints if patience is used
                return self #terminate loop

        #Do not add code between these lines ↑ ↓!
        
        print(f"Terminated at epoch {epoch+1} using self.nb_epoch") # Prints if patience is not used
        
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x, numpy=True):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        
        # one layer (no hidden layer, directly from in to out)
        # calculate forward onto x to get predicted y
        # y_linear = self.model.forward(X)
        # y_activation = self.relu(y_linear)
        
        y_activation = X
        for i in range(len(self.linears)):
            # print("layer: ",i," - ",y_activation.shape)
            y_linear = self.linears[i].forward(y_activation)    # linear transform applied
            if i != 0 and i != len(self.linears)-1:
                y_activation = self.activation(y_linear) # activation in hidden layers
            else:
                y_activation = y_linear # identity
            # print("pass ", i)
        
        if numpy:
            return y_activation.detach().numpy()
        else:
            return y_activation

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1). EXPECTED Y

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, y_expected = self._preprocessor(x, y, training = False) # Do not forget
        
        y_hat = self.predict(X, numpy=False)
        score = self.criterion(y_expected, y_hat)     # Applies the MSE loss function

        return float(score)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    
    def get_params(self, deep=True):
        return {
            'x' : self.x,
            'nb_epoch' : self.nb_epoch,
            'minibatch_size' : self.minibatch_size,
            'lr' : self.lr,
            'optimizer_name' : self.optimizer_name,
            'patience' : self.patience,
            'nb_neurons_per_hidden_layer' : self.nb_neurons_per_hidden_layer
            }

    def set_params(self, **params):
        for (param, value) in params.items():
            setattr(self, param, value)
        return self


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x, y, param): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    '''
    Set params:
    use_patience = False
    epochs = 200
    val_prop = 0.11, #by default it is 9.9% of dataset 
    checked beween sigmoid & relu, and relu is faster
    '''
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # test_size = int(len(x) * 0.1)     #this is not 0.2 as we are also using val; but 0.2 is also possible

    test_fold = np.zeros_like(y)
    test_size = int(len(x) * 0.1)
    test_fold[test_size :] = -1
    splitter = PredefinedSplit(test_fold=test_fold)
    
    grid_search = GridSearchCV(Regressor(x=x, nb_epoch=2000), param_grid=param, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=splitter, verbose = 3, refit=True)

    grid_search.fit(x, y)

    results_df = pd.DataFrame(grid_search.cv_results_)

    print(results_df)

    results_df.to_csv("hyperparameter_results.csv")

    save_regressor(grid_search.best_estimator_)
    
    return  # Return the chosen hyper parameters
    # to find tuned hyper parameters
    # the marking is only testing the model, could find the best parameters for the model

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():
    start_time = datetime.now()
    
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    test_size = int(len(data) * 0.1)     #this is not 0.2 as we are also using val; but 0.2 is also possible
    print("training + val data length: ", int(len(data) - test_size))

    x_data = data.loc[:, data.columns != output_label]
    y_data = data.loc[:, [output_label]]

    x_test = x_data.loc[: test_size]
    y_test = y_data.loc[: test_size]

    x_train_val = x_data.loc[test_size :]
    y_train_val = y_data.loc[test_size :]
    
    # Training

    # Since we use patience, we need val set passed in with train set to match with LabTS format
    regressor = Regressor(x_train_val)

    regressor.fit(x_train_val, y_train_val)
    save_regressor(regressor)

    # Error
    print("Number of neurons per hidden layer: ", regressor.nb_neurons_per_hidden_layer, ", with minibatch size: ", regressor.minibatch_size, " - lr: ", regressor.lr)
    
    MSE = regressor.score(x_test, y_test)
    RMSE = np.sqrt(MSE)
    print(f"Test data MSE: {MSE:.1f}\n")
    print(f"Test data RMSE: {RMSE:.3f}\n")

    ### Dollar error
    # preds = regressor.predict(x_test)
    # __, target = regressor._preprocessor(x_test, y_test, training = False)
    # mean_error = float(abs(torch.mean(target - preds)))
    # median_error = float(abs(torch.median(target - preds)))
    # print(f"median error in $$$: {median_error:.2f}")
    # print(f"mean error in $$$: {mean_error:.2f}")
    ###

    elapsed = datetime.now() - start_time
    print("Time taken: ", elapsed)

def example_main2():
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")
    data = data.sample(frac=1) #shuffle
    x_data = data.loc[:, data.columns != output_label]
    y_data = data.loc[:, [output_label]]

    test_size = int(len(data) * 0.1)     #this is not 0.2 as we are also using val; but 0.2 is also possible
    print("training + val data length: ", int(len(data) - test_size))

    x_data = data.loc[:, data.columns != output_label]
    y_data = data.loc[:, [output_label]]

    x_test = x_data.loc[: test_size]
    y_test = y_data.loc[: test_size]

    x_train_val = x_data.loc[test_size :]
    y_train_val = y_data.loc[test_size :]

    regressor = load_regressor()
    
    print("Number of neurons per hidden layer: ", regressor.nb_neurons_per_hidden_layer, ", with minibatch size: ", regressor.minibatch_size, " - lr: ", regressor.lr)
    
    MSE = regressor.score(x_test, y_test)
    RMSE = np.sqrt(MSE)
    print(f"Test data MSE: {MSE:.1f}\n")
    print(f"Test data RMSE: {RMSE:.3f}\n")

    ### Dollar error
    # preds = regressor.predict(x_test)
    # __, target = regressor._preprocessor(x_test, y_test, training = False)
    # mean_error = float(abs(torch.mean(target - preds)))
    # median_error = float(abs(torch.median(target - preds)))
    # print(f"median error in $$$: {median_error:.2f}")
    # print(f"mean error in $$$: {mean_error:.2f}")
    ###


def hyperparameter_main():

    start_time = datetime.now()

    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")
    x_data = data.loc[:, data.columns != output_label]
    y_data = data.loc[:, [output_label]]
    
    params = {
            "minibatch_size" : [128, 256, 512, 1024],
            "lr" : [0.05, 0.005, 0.0005],
            "optimizer" : [torch.optim.Adam, torch.optim.RMSprop],
            "patience" : [10, 50, 100, 200],
            "nb_neurons_per_hidden_layer" : [
                                            [6],
                                            [12, 4],
                                            [12, 8],
                                            [13, 8, 4],
                                            [8, 4, 3, 2],
                                            [16, 16, 8, 4],
                                            [64, 64, 32],
                                            [32, 32, 16, 8, 4],
                                            [128, 64, 32, 32, 8, 8, 4],
                                            [128, 64, 64, 16, 16, 4, 4]
                                            ]
            }

    RegressorHyperParameterSearch(x_data, y_data, params)

    elapsed = datetime.now() - start_time
    print("Time taken: ", elapsed)


if __name__ == "__main__":
    # example_main()
    # hyperparameter_main()
    example_main2()