import numpy as np
import pickle

np.set_printoptions(precision=5, threshold=30, edgeitems=7)
np.core.arrayprint._line_width = 120
seed = 3408
np.random.seed(seed)

prints = True

def xavier_init(size, gain = 1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        #Treating "_cache_current" as the tuple y_pred, y_target of previous node
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative 
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)      #-∂L/∂y

class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """ 
        Constructor of the Sigmoid layer.
        """
        
        self._cache_current = None

    @staticmethod
    def sigmoid(x):
        return np.reciprocal(1.0 + np.exp(-x))

    def forward(self, x):
        """ 
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################
        
        self._cache_current = x
        return SigmoidLayer.sigmoid(x)
        
        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################
        # sigmoid derivative: g'(z)=g(z)(1-g(z))
        # grad_z = ∂L/∂A input in this layer (but it is the output of ∂L/∂Z in last layer)
        
        dA_dZ = SigmoidLayer.sigmoid(self._cache_current) * (1 - SigmoidLayer.sigmoid(self._cache_current)) #∂A/∂Z
        dL_dZ = np.multiply(grad_z, dA_dZ) # element multiplication                                         #∂L/∂Z
        return dL_dZ

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################
       
        self._cache_current = x
        x[x<0] = 0
        return x

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        ####################################################################### 
        # sigmoid derivative: g'(z)={1 if z>0, 0 otherwise}
        # grad_z = ∂L/∂A input in this layer (but it is the output of ∂L/∂Z in last layer)
        
        dA_dZ = np.where(self._cache_current > 0, 1, 0)                     #∂A/∂Z
        dL_dZ = np.multiply(grad_z, dA_dZ) # element multiplication         #∂L/∂Z
        return dL_dZ

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################
        
        self._W = xavier_init((n_in, n_out))
        self._b = np.zeros((1, n_out)) #(n_out, 1)

        self._cache_current = None  #∂L/∂Z 
        self._grad_W_current = None #∂L/∂W 
        self._grad_b_current = None #∂L/∂b 

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################
        
        self._cache_current = x
        return (x @ self._W) + self._b

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################

        self._grad_W_current = self._cache_current.T @ grad_z                   # ∂L/∂W = ∂L/∂Z * ∂Z/∂W = X * ∂L/∂Z
        
        ones = np.ones((1, grad_z.shape[0]))
        self._grad_b_current = np.squeeze(ones @ grad_z)                                     # ∂L/∂b = ∂L/∂Z * ∂Z/∂b = ∂L/∂Z
        return grad_z @ self._W.T                                               # ∂L/∂X = ∂L/∂Z * ∂Z/∂X = ∂L/∂Z * (W)T

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        ####################################################################### 
        #                       ** START OF YOUR CODE **                      #
        #######################################################################
        
        self._W = self._W - learning_rate*self._grad_W_current
        self._b = self._b - learning_rate*self._grad_b_current
        
        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding 
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################
        
        if prints: print("\n NEW NETWORK")

        self.full_neurons = [self.input_dim]
        self.full_neurons.extend(self.neurons)

        self._layers = []
        for layer_nb in range(len(self.full_neurons)-1):
            layer = LinearLayer(n_in=self.full_neurons[layer_nb], n_out=self.full_neurons[layer_nb+1])
            self._layers.append(layer)
            if activations[layer_nb] == "relu":
                self._layers.append(ReluLayer())
            elif activations[layer_nb] == "sigmoid":
                self._layers.append(SigmoidLayer())

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################

        layer_outputs = x
        for index, layer in enumerate(self._layers):
            linear_out = layer.forward(layer_outputs)
            layer_outputs = linear_out

        return linear_out

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################

        layer_outputs = grad_z
        for index, layer in enumerate(reversed(self._layers)):
            output = layer.backward(layer_outputs)
            layer_outputs = output

        return layer_outputs       #∂Loss/∂X

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################

        for index, layer in enumerate(reversed(self._layers)):
            # update_params per layer
            if type(layer) is LinearLayer:
                layer.update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                cross_entropy.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################

        if self.loss_fun == 'mse':
            self._loss_layer = MSELossLayer()
        elif self.loss_fun == 'cross_entropy':
            self._loss_layer = CrossEntropyLossLayer()

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns: 
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################
        
        shuffled_indices = np.random.permutation(len(input_dataset))
        shuffled_inputs = input_dataset[shuffled_indices]
        shuffled_targets =  target_dataset[shuffled_indices]
        
        return shuffled_inputs, shuffled_targets

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      # 
        #######################################################################
        
        global seed
        if prints: print("\n TRAINING")
        
        for epoch in range(self.nb_epoch):
            if self.shuffle_flag:
                input_dataset, target_dataset = Trainer.shuffle(input_dataset, target_dataset)

            if len(input_dataset) % self.batch_size:
                split_indices = [(i+1)*self.batch_size for i in range(int(np.floor(len(input_dataset) / self.batch_size)))]
            else:
                split_indices = [(i+1)*self.batch_size for i in range(int(np.floor(len(input_dataset) / self.batch_size))-1)]

            input_batches = np.array_split(input_dataset, split_indices)
            target_batches = np.array_split(target_dataset, split_indices)
            if not epoch % 100:
                if prints: print("batch sizes: ", [len(batch) for batch in input_batches])
            
            for input_batch, target_batch in zip(input_batches, target_batches):
                computed_loss = self.eval_loss(input_batch, target_batch)       # Contains forward pass and Loss function

                dL_dy = self._loss_layer.backward()
                
                backprop_grad_wrt_inputs = self.network.backward(dL_dy)         # start backpropagation with first grad_z being the loss function
                
                self.network.update_params(self.learning_rate)      # APPLY GRADIENT DECENT ALGORITHM, MINI BATCHED GRADIENT DECENT
            
            if not epoch % 100:
                if prints: print(f"epoch {epoch} training loss: ", computed_loss)
                
        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data. Returns
        scalar value.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).
        
        Returns:
            a scalar value -- the loss
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################     

        fwd_pass_outputs = self.network.forward(input_dataset)                  # initial forward pass
        loss = self._loss_layer.forward(fwd_pass_outputs, target_dataset)

        return loss

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################
       
        self.a = 0
        self.b = 1

        if prints: print("preprocessing data of size: ", data.shape)

        self.xmin = []
        self.xmax = []

        for col in data.T: #for each feature
            self.xmin.append(np.amin(col))   #store smallest
            self.xmax.append(np.amax(col))    #and largest for each feature

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################

        norm_data = np.zeros_like(data.T)
        
        for i, col in enumerate(data.T):
            if (self.xmax[i] - self.xmin[i]) != 0: #division by 0 error from no range
                col = self.a + (col - self.xmin[i])*(self.b - self.a) / (self.xmax[i] - self.xmin[i]) 
            norm_data[i] = col

        return norm_data.T

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retrieve the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **                      #
        #######################################################################

        revert_data = np.zeros_like(data.T)
        
        for i, col in enumerate(data.T):
            col = self.xmin[i] + (col - self.a)*(self.xmax[i] - self.xmin[i]) / (self.b - self.a)
            revert_data[i] = col
            
        return revert_data.T

        #######################################################################
        #                       ** END OF YOUR CODE **                        #
        #######################################################################

def example_main():
    global prints
    prints = True
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.001,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


def example_main2 ():
    global prints
    prints = True
    input_dim = 13
    neurons = [16, 16, 8, 4, 1]
    activations = ["relu", "relu", "relu", "relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("housing.txt")

    x = dat[:, :-1]
    y = dat[:, -1]
    y = y[:, np.newaxis]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=256,
        nb_epoch=1000,
        learning_rate=0.000001,
        loss_fun="mse",
        shuffle_flag=False,
    )

    trainer.train(x_train_pre, y_train)

    print("Regression Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Regression Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    save_network(net, "part_1_save_test.pickle")
    net_loaded = load_network("part_1_save_test.pickle")

    preds = net_loaded.forward(x_train_pre)                  #initial forward pass
    loss = trainer._loss_layer.forward(preds, y_train)
    print("Regression Loaded RMSE loss = ", np.sqrt(loss))

if __name__ == "__main__":
    example_main()
    # example_main2()

