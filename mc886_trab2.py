import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
##
##          Lendo e preparando o Dataset
##

def normalize_features(features, mean, std):
    output = np.copy(features)
    output = np.subtract(output, mean)
    output = np.divide(output, std)
    return output


# dataset = pd.read_csv('fashion-mnist_train.csv')
# dataset = dataset.sample(frac=1)
# dataset = np.array(dataset)
# train_set    = dataset[0:50000,  2:]
# valid_set    = dataset[50000:,2:]
# train_labels = dataset[0:50000,1]
# valid_labels = dataset[50000:,1]
# np.savetxt('fashion-mnist_valid-set.csv', valid_set, delimiter=',',fmt='%i')
# np.savetxt('fashion-mnist_train-labels.csv', train_labels, delimiter=',',fmt='%i')
# np.savetxt('fashion-mnist_valid-labels.csv', valid_labels, delimiter=',',fmt='%i')
# np.savetxt('fashion-mnist_train-set.csv', train_set, delimiter=',',fmt='%i')


train_set    = np.genfromtxt('fashion-mnist_train-set.csv', delimiter=',')
valid_set    = np.genfromtxt('fashion-mnist_valid-set.csv', delimiter=',')
train_labels = np.genfromtxt('fashion-mnist_train-labels.csv', delimiter=',')
valid_labels = np.genfromtxt('fashion-mnist_valid-labels.csv', delimiter=',')


train_set = train_set.astype(int)
valid_set = valid_set.astype(int)
train_labels = train_labels.astype(int)
valid_labels = valid_labels.astype(int)


mean = np.mean(train_set, axis=0)
std = np.std(train_set, axis=0)
train_set_n = normalize_features(train_set, mean, std)
valid_set_n = normalize_features(valid_set, mean, std)

# img = Image.fromarray(train_set[8].reshape(28, 28).astype('uint8')*255)
# img.show()


##
##          Definindo funcoes para a regressao e rede neural
##

##
## Regressao logistica
##

def sigmoid(z):
    g = np.multiply(z, -1)
    g = np.exp(g)
    g = np.add(g, 1)
    g = np.true_divide(1,g)
    g[g == 1] = 0.9999
    g[g == 0] = 0.0001
    return g

def train_lr(theta, X, Y, iteracoes, alpha):
    dash = '-' * 40
    costs = []
    n = len(theta)
    m = len(X)
    Xt = np.transpose(X)
    grad = alpha*(1/m)

    j=0
    for i in range(0,iteracoes):
        H = sigmoid(np.dot(X, theta))
        loss = np.mean(np.subtract(np.multiply(np.multiply(-1, Y),np.log(H)),np.multiply(np.subtract(1, Y),np.log(np.subtract(1,H)))))
        #loss = (1 / m) * (-Y.T.dot(np.log(H)) - (1 - Y).T.dot(np.log(1 - H)))
        #print(loss)
        J = np.subtract(H,Y)
        J = np.dot(Xt, J)
        new_theta = np.zeros(n)
        J = np.multiply(grad,J)
        new_theta = np.subtract(theta,J)
        theta = new_theta
        costs.append(loss)
        if j==9:
            print("Iteração: "+str(i+1)+" loss:"+str(loss))
            j=0
        j+=1

    predict = sigmoid(np.dot(X, theta))
    predict[predict >= 0.5] = 1
    predict[predict < 0.5] = 0
    hits = np.sum(predict == Y)
    accuracy = hits/len(Y)
    print("Acurácia: "+str(accuracy))
    plt.plot(range(0,iteracoes),costs)
    plt.ylabel('Custo')
    plt.xlabel('Iterações')
    plt.show()
##
## Executando regressao logistica
##


theta = np.random.rand(train_set_n.shape[1])
y = np.zeros((train_labels.shape))
y[train_labels == 0] = 1
y[train_labels != 0] = 0
train_lr(theta, train_set_n, y, 1000, 0.02)
theta_class0 = theta


##
## Rede neural
##


class Layer:
    def __init__(self, input_size, layer_size, activation_function, derivative_function):
        self.weights = np.random.rand(input_size, layer_size)
        self.activation = activation_function
        self.derivative = derivative_function

    def forward(self, layer_input, activation_function):
        z = np.dot(layer_input, self.weights)
        return activation_function(z)

    def backprop(self, next_layer_error, derivative_function, learning_rate):
        self.error = np.dot(np.transpose(next_layer_error), self.weights)
        self.delta = self.error * derivative_function(self.activation)
        self.weights += self.activation.T.dot(next_layer_error)

def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

def relu(x):
    return np.max(x,0)

def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sigmoidDerivative(x):
    div = np.multiply(x, -1)
    div = np.exp(div)
    g = np.add(div, 1)
    g = np.power(g,2)
    g = np.true_divide(div,g)
    return g

def softmax(x):
    s = np.exp(x)
    s = np.true_divide(s,np.sum(s))
    return s

#def softmaxDerivative(x):


##
## Executando a rede neural
##

neural_net_logistic = []

# Adicionando a camada de input no indice 0
neural_net_logistic.append(train_set_n)

# Adicionando a camada de pesos
logistic_parameters = Layer(neural_net_logistic[0].shape[1], 1, sigmoid, sigmoidDerivative)
neural_net_logistic.append(logistic_parameters)

# Adicionando a camada de saída
neural_net_logistic.append(neural_net_logistic[1].forward(neural_net_logistic[0], sigmoid))


def backpropagation(neural_net, y):
    total_error = y - neural_net[-1]
    total_error = total_error*sigmoidDerivative(neural_net[-1])
    for i in reversed(range(0, len(neural_net))):
        print(i)

backpropagation(neural_net_logistic, train_labels)

def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights
