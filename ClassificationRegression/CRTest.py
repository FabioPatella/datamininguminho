import unittest
import numpy as np
from scipy.optimize import minimize

from DeepNN import DeepNeuralNetwork
from SimpleNN import NeuralNetwork


class CRTest(unittest.TestCase):
    def testSimpleNN(self):
        input_size = 2
        hidden_size = 3
        output_size = 1

        nn = NeuralNetwork(input_size, hidden_size, output_size)

        # define input and target data
        nn.input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        nn.target_data = np.array([[0], [1], [1], [0]])

        # define initial weights
        initial_weights = np.concatenate((nn.get_input_weights().ravel(), nn.get_output_weights().ravel()))
        # call minimize to optimize the weights
        result = minimize(nn.cost_function, initial_weights, method='BFGS')
        # retrieve the optimized weights and update the neural network
        optimized_weights = result.x
        nn.input_weights = optimized_weights[:nn.input_weights.size].reshape(nn.input_weights.shape)
        nn.output_weights = optimized_weights[nn.input_weights.size:].reshape(nn.output_weights.shape)
        predictions=nn.predict(nn.input_data)
        self.assertGreaterEqual(predictions[0], -0.1)
        self.assertLessEqual(predictions[0], 0.2)
        self.assertGreaterEqual(predictions[1], 0.9)
        self.assertLessEqual(predictions[1], 1.1)
    def testDeepNN(self):
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
        for i in range(nn.num_layers - 1):
            end_index = start_index + nn.weights[i].size
            nn.weights[i] = optimized_weights[start_index:end_index].reshape(nn.weights[i].shape)
            start_index = end_index
        predictions = nn.predict(nn.input_data)
        self.assertGreaterEqual(predictions[0], -0.1)
        self.assertLessEqual(predictions[0], 0.2)
        self.assertGreaterEqual(predictions[1], 0.9)
        self.assertLessEqual(predictions[1], 1.1)

if __name__ == '__main__':
    unittest.main()
