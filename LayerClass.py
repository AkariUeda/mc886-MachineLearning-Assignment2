import numpy as np
from functions import *
import matplotlib.pyplot as plt

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
        self.train_loss = []
        self.valid_loss = []

    def calc_loss(self,H,y,group):
        Y = np.zeros((len(y),10))
        for i in range(0, len(y)):
            Y[i][y[i]] = 1
        out = len(self.camadas)-1
        m = H.shape[0]
        if group == 'train':
            if self.functions[out] == sigmoid:
                self.train_loss.append(np.sum(-1* np.add(np.multiply(y,(1/H)) , np.multiply(np.subtract(1, y),(1/np.subtract(1,H))))))
            elif self.functions[out] == softmax:
                self.train_loss.append((-1 / m) * np.sum(Y * np.log(H)) + (1/2)*np.sum(self.camadas[out].weights**2))
                #self.train_loss.append(np.sum(-np.log(H[Y])))
        elif group == 'valid':
            if self.functions[out] == sigmoid:
                self.valid_loss.append(np.sum(-1* np.add(np.multiply(y,(1/H)) , np.multiply(np.subtract(1, y),(1/np.subtract(1,H))))))
            elif self.functions[out] == softmax:
                #self.valid_loss.append(np.sum(-np.log(H[Y])))
                self.valid_loss.append( (-1 / m) * np.sum(Y * np.log(H)) + (1/2)*np.sum(self.camadas[out].weights**2))
        return 

    def forward(self,X,y):
        out = len(self.camadas)-1
        inp = 0
        self.camadas[inp].activation = X
        #print(self.camadas[inp].activation[0])
        for i in range(1,len(self.camadas)):
            self.camadas[i].activation = self.functions[i](np.add(self.camadas[i-1].activation.dot(self.camadas[i].weights),self.camadas[i].bias.T))
            #self.camadas[i].activation = self.functions[i](self.camadas[i-1].activation.dot(self.camadas[i].weights))
        self.calc_loss(self.camadas[out].activation, y, 'train')
        return self.camadas[out].activation

    def forward_pred(self,activation,y):
        for i in range(1,len(self.camadas)):
            activation = self.functions[i](np.add(activation.dot(self.camadas[i].weights),self.camadas[i].bias.T))
        self.calc_loss(activation, y, 'valid')
        return activation

    def backward(self,  X, y, learning_rate):
        out = len(self.camadas)-1
        inp = 0
        #print(self.camadas[out].weights[0])
        if self.functions[out] == softmax:
            Y = np.zeros((len(y),10))
            for i in range(0, len(y)):
                Y[i][y[i]] = 1
            #print(Y.shape)
            
            self.camadas[out].delta = np.subtract(Y,self.camadas[out].activation)
    
        if self.functions[out] == sigmoid:
            self.camadas[out].error = y-self.camadas[out].activation
            self.camadas[out].delta = self.camadas[out].error*self.derivatives[out](self.camadas[out].activation)
            
        for i in range(len(self.camadas)-2,0,-1):
            self.camadas[i].error = self.camadas[i+1].delta.dot(self.camadas[i+1].weights.T)
            self.camadas[i].delta = self.camadas[i].error*self.derivatives[i](self.camadas[i].activation)
            
            #self.camadas[i].delta = delta_cross_entropy(self.camadas[i].activation,y)*self.derivatives[i](self.camadas[i].activation)

        for i in range(len(self.camadas)-1,0,-1):
            w = learning_rate*self.camadas[i-1].activation.T.dot(self.camadas[i].delta)
            #print(str(self.camadas[i].delta[0][0])+" * "+str(self.camadas[i-1].activation.T[0][0]))
            #print(self.camadas[i].weights)
            self.camadas[i].weights += w
            self.camadas[i].bias += np.sum(self.camadas[i].delta) * learning_rate


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
        preds[preds <= 0.5] = 0
        acc = sum(preds == y)
        print("Acurácia validação: "+str(acc/len(y)))

    def train_onevsall(self,X,y,learning_rate,iteracoes, printacc):
        acc = 0
        for i in range(0, iteracoes):
            self.forward(X,y)
            self.backward(X,y,learning_rate)
            preds = np.copy(self.camadas[1].activation)
            preds[preds > 0.5] = 1
            preds[preds <=0.5] = 0
            acc = sum(preds == y)/len(y)
            if printacc:               
                print("Acc: "+str(acc))
        return acc

    def train_neuralnet(self,X,y, Xv, yv,learning_rate,iteracoes, printacc):
        acc = 0
        for i in range(0, iteracoes):
            p_train = self.forward(X,y)
            p_valid = self.forward_pred(Xv,yv)
            self.backward(X,y,learning_rate)
            p_train = np.argmax(p_train, axis=1)
            p_valid = np.argmax(p_valid, axis=1)
            y1 = y.reshape((y.shape[0]))
            yv1 = yv.reshape((yv.shape[0]))
            acc_train = sum(p_train == y1)/len(y1)
            acc_valid = sum(p_valid == yv1)/len(yv1)
            if printacc:               
                print("Acc treino: "+str(acc_train))
                print("Acc valid: "+str(acc_valid))
        plt.plot(range(0,iteracoes),self.train_loss,  'r-', label='Train')
        plt.plot( range(0,iteracoes), self.valid_loss, 'g-', label='Valid')
        plt.title('title')
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.legend()
        plt.savefig('treino.png')
        plt.show()
        return acc

    def predict_prob(self,X,y):
        camadas = np.copy(self.camadas)
        out = len(camadas)-1
        inp = 0
        camadas[0].activation = X
        for i in range(1,len(camadas)):
            camadas[i].activation = self.functions[i](camadas[i-1].activation.dot(camadas[i].weights))
        preds = camadas[out].activation
        return preds
        
        
        
