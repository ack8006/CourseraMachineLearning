from __future__ import division
import numpy as np
import scipy
import scipy.io
from matplotlib import pyplot, cm
import copy



class NeuralNetwork(object):
    def __init__(self, sizes, X, y):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #self.y.shape == (5000,10)
        self.y = self.convert_y(y, sizes[self.num_layers-1])
        #self.y.shape == (5000,400)
        self.X = X
        print self.y.shape
        print self.X.shape

        self.thetas = []
        #self.thetas[0].shape == (25,401)
        #self.thetas[1].shape == (10x26)
        for x in xrange(len(sizes)-1):
            self.thetas.append(self.randInitialThetas(sizes[x], sizes[x+1]))

        #print self.compute_cost()
        #print self.feed_forward_full()[0]

    def convert_y(self, y, output_size):
        m = np.shape(y)[0]
        output = np.zeros((m, output_size))
        for a in xrange(m):
            if y[a] == 10:
                output[a, 0] = 1
            else:
                output[a, y[a]] = 1
        return output

    def compute_cost(self):
        m = len(self.y)
        a = np.copy(self.X)
        for theta in self.thetas:
            z = self.feed_forward(a, theta)
            a = sigmoid(z)
        inner = (-self.y * np.log(a)) - (1-self.y)*np.log(1-a)
        J = (1/m) * sum(sum(inner))
        return J

    def regularize_cost(self, lamb, m):
        return sum(map(lambda x:sum(sum(np.square(x[:,1:]))), self.thetas))

    #takes all inputs
    #***RETURNS Z NOT A, MUST BE SIGMOIDED
    def feed_forward(self, a, theta):
        a = np.hstack([np.ones((len(a),1)), a])
        return a.dot(theta.T)

    def feed_forward_full(self):
        m = len(self.y)
        a = np.copy(self.X)
        for theta in self.thetas:
            z = self.feed_forward(a, theta)
            a = sigmoid(z)
        return a

    def randInitialThetas(self, num_input_layers, num_output_layers):
        #epsilon
        e = 0.12
        return np.random.random((num_output_layers, num_input_layers+1))*2*e-e

    def backprop(self):
        a_output = self.feed_forward_full()





def sigmoid(z):
    #return np.divide(1.0, (1.0 + np.exp(-z)))
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_gradient(z):
    #return np.muliply(sigmoid(z),(1-sigmoid(z)))
    return sigmoid(z)*(1-sigmoid(z))

def part_2():
    data = scipy.io.loadmat('ex4data1.mat')
    X,y = data['X'], data['y']
    input_later = 400
    hidden_layer = 25
    output_layer = 10
    sizes = [input_later, hidden_layer, output_layer]

    nn = NeuralNetwork(sizes,X,y)


if __name__ == '__main__':
    part_2()








