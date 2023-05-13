import numpy as np
from scipy.optimize import minimize

class DeepNeuralNetwork:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.normal(0, 1, (layer_sizes[i], layer_sizes[i+1])) for i in range(self.num_layers-1)]
        self.input_data = None
        self.target_data = None

    def sigmoid(self, x):
        # sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def predict(self, input_data):
        # forward pass through the network
        hidden_layer = input_data
        for i in range(self.num_layers-2):
            hidden_layer = self.sigmoid(np.dot(hidden_layer, self.weights[i]))
        output_layer = np.dot(hidden_layer, self.weights[-1])
        return output_layer

    def cost_function(self, weights):
        # mean squared error cost function
        start_index = 0
        for i in range(self.num_layers-1):
            end_index = start_index + self.weights[i].size
            self.weights[i] = weights[start_index:end_index].reshape(self.weights[i].shape)
            start_index = end_index
        predicted_data = self.predict(self.input_data)
        return np.mean(np.square(self.target_data - predicted_data))

    def get_weights(self):
        return self.weights
if __name__ == '__main__':

 layer_sizes = [2, 3, 1]

 nn = DeepNeuralNetwork(layer_sizes)

 # define input and target data
 nn.input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
 nn.target_data = np.array([[0], [1], [1], [0]])

 # define initial weights
 initial_weights = np.concatenate([w.ravel() for w in nn.get_weights()])

 print("testing prediction before optimization")
 print(nn.predict(nn.input_data))

 # call minimize to optimize the weights
 result = minimize(nn.cost_function, initial_weights, method='BFGS')

 # retrieve the optimized weights and update the neural network
 optimized_weights = result.x
 start_index = 0
 for i in range(nn.num_layers-1):
    end_index = start_index + nn.weights[i].size
    nn.weights[i] = optimized_weights[start_index:end_index].reshape(nn.weights[i].shape)
    start_index = end_index

 print("testing prediction after optimization")
 print(nn.predict(nn.input_data))
