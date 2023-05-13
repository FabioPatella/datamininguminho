import numpy as np
from scipy.optimize import minimize

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # initialize weights randomly with mean 0
        self.input_weights = np.random.normal(0, 1, (input_size,hidden_size))
        self.output_weights = np.random.normal(0, 1, (hidden_size,output_size))
        self.input_data = None
        self.target_data = None

    def sigmoid(self, x):
        # sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def predict(self, input_data):
        # forward pass through the network
        hidden_layer = self.sigmoid(np.dot(input_data, self.input_weights))
        output_layer = np.dot(hidden_layer, self.output_weights)
        return output_layer

    def cost_function(self, weights):
        # mean squared error cost function
        self.input_weights = weights[:self.input_weights.size].reshape(self.input_weights.shape)
        self.output_weights = weights[self.input_weights.size:].reshape(self.output_weights.shape)
        predicted_data = self.predict(self.input_data)
        return np.mean(np.square(self.target_data - predicted_data))

    def get_input_weights(self):
        return self.input_weights

    def get_output_weights(self):
        return self.output_weights
if __name__ == '__main__':

 input_size = 2
 hidden_size = 3
 output_size = 1

 nn = NeuralNetwork(input_size, hidden_size, output_size)

 # define input and target data
 nn.input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
 nn.target_data = np.array([[0], [1], [1], [0]])

 # define initial weights
 initial_weights = np.concatenate((nn.get_input_weights().ravel(), nn.get_output_weights().ravel()))

 print("testing prediction before optimization")
 print(nn.predict(nn.input_data))
 # call minimize to optimize the weights
 result = minimize(nn.cost_function, initial_weights, method='BFGS')
 # retrieve the optimized weights and update the neural network
 optimized_weights = result.x
 nn.input_weights = optimized_weights[:nn.input_weights.size].reshape(nn.input_weights.shape)
 nn.output_weights = optimized_weights[nn.input_weights.size:].reshape(nn.output_weights.shape)

 # test the neural network
 print("testing prediction after optimization")
 print(nn.predict(nn.input_data))
