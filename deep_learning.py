
######################## Multiple Layer Perceptron ######################

import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Parametrlarni initsializatsiya qilish
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
    
    def compute_loss(self, y, y_pred):
        m = y.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y.flatten()])
        loss = np.sum(log_likelihood) / m
        return loss

# Ma'lumotlar yaratish
np.random.seed(42)
X = np.random.randn(100, 3)
y = np.random.randint(0, 2, size=(100, 1))

# Model yaratish va oldinga siljish
mlp = MLP(input_size=3, hidden_size=5, output_size=2)
y_pred = mlp.forward(X)

# Yo‘qotishni hisoblash
loss = mlp.compute_loss(y, y_pred)
print(f'Loss: {loss}')


############################### ReLU activation function #############################

def relu(Z):
    return np.maximum(0, Z)

# Test qilish
Z = np.array([[1, -2, 3], [-1, 2, -3]])
print("ReLU natijasi:")
print(relu(Z))

############################## Softmax function ######################################

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# Test qilish
Z = np.array([[1, 2, 3], [1, 2, 3]])
print("Softmax natijasi:")
print(softmax(Z))

############################# Oldinga va Ortga Siljish Algoritmi  #############################

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = softmax(Z2)
    return A1, A2

def backward_propagation(X, y, A1, A2, W2):
    m = X.shape[0]
    dZ2 = A2
    dZ2[range(m), y.flatten()] -= 1
    dZ2 /= m

    dW2 = A1.T.dot(dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * (A1 > 0)
    dW1 = X.T.dot(dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# Model yaratish
input_size, hidden_size, output_size = 3, 5, 2
X = np.random.randn(100, input_size)
y = np.random.randint(0, output_size, size=(100, 1))
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

A1, A2 = forward_propagation(X, W1, b1, W2, b2)
dW1, db1, dW2, db2 = backward_propagation(X, y, A1, A2, W2)

print("Gradients:")
print("dW1:", dW1)
print("db1:", db1)
print("dW2:", dW2)
print("db2:", db2)


############################## CNN Arxitekturasi va Ishlash Strukturasi ############################

from scipy.signal import convolve2d

def conv2d(X, K):
    return convolve2d(X, K, mode='valid')

def relu(X):
    return np.maximum(0, X)

def max_pooling(X, size=2):
    return X[::size, ::size]

# Misol ma'lumotlar va kernel
X = np.array([[1, 2, 3, 0],
              [4, 5, 6, 0],
              [7, 8, 9, 0],
              [0, 0, 0, 0]])

K = np.array([[1, 0],
              [0, -1]])

# Konvolyutsiya va max pooling
conv_output = conv2d(X, K)
relu_output = relu(conv_output)
pool_output = max_pooling(relu_output)

print("Konvolyutsiya natijasi:")
print(conv_output)
print("ReLU natijasi:")
print(relu_output)
print("Max Pooling natijasi:")
print(pool_output)

########################  RNN  ########################

class VanillaRNN:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, 1) * 0.01
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, 1))
    
    def forward(self, X):
        h = np.zeros((X.shape[0], self.hidden_size))
        for t in range(X.shape[1]):
            h = np.tanh(np.dot(X[:, t, :], self.Wxh) + np.dot(h, self.Whh) + self.bh)
        y = np.dot(h, self.Why) + self.by
        return y

# Test qilish
X = np.random.randn(10, 5, 3)  # (batch_size, sequence_length, input_size)
rnn = VanillaRNN(input_size=3, hidden_size=4)
y_pred = rnn.forward(X)
print("Vanilla RNN chiqishi:")
print(y_pred)
    
