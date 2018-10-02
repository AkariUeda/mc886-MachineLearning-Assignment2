import numpy as np
from functions import *

class Layer:
    def __init__(self, random, input_size, output_size):
        self.activation = np.array((output_size,1))

        if random == True:
            self.weights = np.random.uniform(-0.05,0.05,(input_size, output_size))
            self.bias = np.random.uniform(-0.05,0.05,(output_size,1))
        else:
            self.weights = np.identity(input_size)
            self.bias = np.zeros(output_size)


class NeuralNetwork:
    def __init__(self):
        self.camadas = []
        self.functions = []
        self.derivatives = []

    def loss(self,H,Y):
        return np.mean(np.subtract(np.multiply(np.multiply(-1, Y),np.log(H)),np.multiply(np.subtract(1, Y),np.log(np.subtract(1,H)))))

    def forward(self,X):
        out = len(self.camadas)-1
        inp = 0
        self.camadas[inp].activation = X
        for i in range(1,len(self.camadas)):
            #self.camadas[i].activation = self.functions[i](self.camadas[i-1].activation.dot(self.camadas[i].weights))+self.camadas[i].bias]
            self.camadas[i].activation = self.functions[i](self.camadas[i-1].activation.dot(self.camadas[i].weights))
        return self.camadas[out].activation

    def backward(self,  X, y, learning_rate):
        out = len(self.camadas)-1
        inp = 0
        self.camadas[out].error = cross_entropy(self.camadas[out].activation,y)
        self.camadas[out].delta = self.camadas[out].error*self.derivatives[out](self.camadas[out].activation, y)

        for i in range(len(self.camadas)-2,0,-1):
            print(i)
            self.camadas[i].error = self.camadas[i+1].delta.dot(self.camadas[i].weights)
            self.camadas[i].delta = self.camadas[i].error*self.derivatives[i](self.camadas[i].activation)
        for i in range(len(self.camadas)-1,0,-1):
            self.camadas[i].weights += learning_rate*self.camadas[i-1].activation.T.dot(self.camadas[i].delta)

    def predict(self,X,y):
        camadas = np.copy(self.camadas)
        out = len(camadas)-1
        inp = 0
        camadas[0].activation = X
        for i in range(1,len(camadas)):
            camadas[i].activation = sigmoid(camadas[i-1].activation.dot(camadas[i].weights))
        output = camadas[out].activation
        preds = camadas[out].activation
        preds[preds > 0.5] = 1
        preds[preds <=0.5] = 0
        acc = sum(preds == y)
        print("Acurácia validação: "+str(acc/len(y)))

    def train(self,X,y,learning_rate,iteracoes, printacc):
        acc = 0
        for i in range(0, iteracoes):
<<<<<<< HEAD
            self.forward(X)
            self.backward(X,y,learning_rate)
            preds = np.copy(self.camadas[1].activation)
            preds[preds > 0.5] = 1
            preds[preds <=0.5] = 0
            acc = sum(preds == y)/len(y)
            if printacc:               
                print("Acc: "+str(acc))
        return acc
=======
            output = self.forward(X)
            if printacc:
                preds = np.copy(output)
                print("Loss: "+str(cross_entropy(output, y)))
                preds[preds > 0.5] = 1
                preds[preds <=0.5] = 0
                acc = sum(preds == y)
                print("Acc: "+str(acc/len(y)))
            self.backward(X,y,learning_rate)
>>>>>>> 59b3c5e60f01c1e33619e33e1c80361ee4dbfa78

    def predict_prob(self,X,y):
        camadas = np.copy(self.camadas)
        out = len(camadas)-1
        inp = 0
        camadas[0].activation = X
        for i in range(1,len(camadas)):
            camadas[i].activation = self.functions[i](camadas[i-1].activation.dot(camadas[i].weights))
        preds = camadas[out].activation
        #print("Pred do predict")
        #print(preds)
        return preds
