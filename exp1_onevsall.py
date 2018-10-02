import numpy as np
import math
import pandas as pd
from LayerClass import Layer, NeuralNetwork
from random import randrange, uniform
from functions import *
import get_dataset

##
## ONE VS ALL
##

class OneVsAllClassifier:
    def __init__(self, classes):
        self.classes = classes
        
    def train(self,X,yl,lr,it):        
        self.neural_net = []
        for i in range(self.classes):
            self.neural_net.append(NeuralNetwork())
        #y = np.zeros((yl.shape[0],self.classes))
        y = np.repeat(yl,10,axis=1)
        y = y.T
      
        for i in range(self.classes):
            print("Forma do Y[i], i = "+str(i))
            print(y[i].shape)
            nc = y[i].reshape((400, 1))
            nc[nc == i] = 1
            nc[nc != i] = 0
            print("NC shape 0")
            print(nc.shape)
            # Adicionando a camada de input no indice 0
            
            self.neural_net[i].camadas.append(Layer(False, X.shape[1], X.shape[1]))
            self.neural_net[i].functions.append(identidade) 
            self.neural_net[i].derivatives.append(identidade)
            self.neural_net[i].forward(X)
            #print("Primeira camada adicionada")

            # Adicionando a camada de saida no indice 1
            self.neural_net[i].camadas.append(Layer(True,self.neural_net[i].camadas[0].activation.shape[1], 1 ))
            self.neural_net[i].functions.append(sigmoid)
            self.neural_net[i].derivatives.append(sigmoidDerivative)
            #print("Segunda camada adicionada") 
            self.neural_net[i].train(X,nc,lr,it,False)

    def classify(self,X,y):
        probs = [];
        for i in range(self.classes):
            probs.append(self.neural_net[i].predict_prob(X,y))
        print(probs)
            

def main():
    train_set, valid_set, train_labels, valid_labels = get_dataset.main()
    cl = OneVsAllClassifier(10)
    cl.train(train_set,train_labels,0.002,10)
    cl.classify(valid_set,valid_labels)
    
if __name__ == "__main__":
    main()


