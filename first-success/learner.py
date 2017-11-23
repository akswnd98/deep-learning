import numpy as np
import matplotlib.pylab as plt
from collections import OrderedDict
import pickle

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        y = np.dot(x, self.W) + self.b

        return y

    def backward(self, dy):
        dx = np.dot(dy, self.W.T)
        self.dW = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0)

        return dx
    
class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y

        return y

    def backward(self, dy):
        return dy * (1.0 - self.y) * self.y

def softmax(x):
    exp_x = np.exp(x - np.array([np.max(x, axis=1)]).T)

    return exp_x / np.array([np.sum(exp_x, axis=1)]).T

def cross_entropy_error(y, t):
    batch_size = y.shape[0]
    
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)

        return self.loss

    def backward(self, dy=1):
        dx = self.y
        batch_size = dx.shape[0]
        for i in range(batch_size):
            dx[i][self.t[i]] -= 1
        dx /= batch_size

        return dx
        
class Network:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        return self.lastLayer.forward(self.predict(x), t)

    def gradient(self, x, t):
        self.loss(x, t)

        dy = 1
        dy = self.lastLayer.backward(dy)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dy = layer.backward(dy)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

if __name__ == '__main__':
    f = open('train-images.idx3-ubyte', 'rb')
    f.seek(16, 0)
    x = np.empty((60000, 784), dtype='float64')
    for i in range(60000):
        x[i] = np.float64(np.frombuffer(f.read(784), dtype='uint8')) / 256
    f.close()

    f = open('train-labels.idx1-ubyte', 'rb')
    f.seek(8, 0)
    t = np.empty(60000, dtype='int')
    for i in range(60000):
        t[i] = np.int(np.frombuffer(f.read(1), dtype='uint8'))
    f.close()

    network = Network(784, 50, 10)

    loss_list = []
    for i in range(10000):
        batch_mask = np.random.choice(60000, 100)
        x_mini = x[batch_mask]
        t_mini = t[batch_mask]

        grad = network.gradient(x_mini, t_mini)
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= 0.1 * grad[key]

        loss = network.loss(x_mini, t_mini)
        loss_list += [loss]
        print(i)

    f = open('params.dat', 'wb+')
    pickle.dump(network, f)
    f.close()

    plt.plot(range(10000), loss_list)
    plt.show()


    f = open('t10k-images.idx3-ubyte', 'rb')
    f.seek(16, 0)
    x = np.empty((10000, 784), dtype='float64')
    for i in range(10000):
        x[i] = np.float64(np.frombuffer(f.read(784), dtype='uint8')) / 256
    f.close()

    f = open('t10k-labels.idx1-ubyte', 'rb')
    f.seek(8, 0)
    t = np.empty(10000, dtype='int')
    for i in range(10000):
        t[i] = np.int(np.frombuffer(f.read(1), dtype='uint8'))
    f.close()

