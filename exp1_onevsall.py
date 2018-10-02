import numpy as np
import math
import pandas as pd
from LayerClass import Layer, NeuralNetwork
from random import randrange, uniform
from functions import *
import get_dataset



##
## REGRESSAO LOGISTICA COM REDE NEURAL DE UMA CAMADA
##    

neural_net_logistic = NeuralNetwork()
y = np.zeros((train_labels.shape))
y[train_labels == 0] = 1
y[train_labels != 0] = 0

# Adicionando a camada de input no indice 0
neural_softmax.camadas.append(Layer(False, train_set.shape[1], train_set.shape[1]))
neural_softmax.functions.append(identidade)
neural_softmax.forward(train_set)

# Adicionando a camada de saída no índice 1
neural_net_logistic.camadas.append(Layer(True, neural_net_logistic.camadas[0].activation.shape[1], 1 ))
print("Segunda camada adicionada")


##
## ONE VS ALL
##

class OneVsAllClassifier:
    def __init__(self, classes):
        self.classes = classes
        
    def train(self,X,y,lr):        
        self.neural_net = []*self.classes
        self.neural_net = []
        y = np.zeros((train_labels.shape))*self.classes
        for i in range(self.classes):
            y[i][train_labels == i] = 1
            y[i][train_labels != i] = 0
            # Adicionando a camada de input no indice 0
            neural_net[i].append(Layer(False, train_set.shape[1], train_set.shape[1]))
            neural_net[i][0].forward(train_set,identidade)
            print("Primeira camada adicionada")

            # Adicionando a camada de saída no índice 1
            neural_net[i].append(Layer(True, neural_net[i][0].activation.shape[1], 1 ))
            print("Segunda camada adicionada")  

    def classify(X,y):
        bestClass = -1
        bestProb = -1
        for i in range(self.classes):
            prob = predict(self.neural_net[i],X,y)
            if prob > bestProb:
                bestClass = i
                bestProb = prob

def main():
    train_set, valid_set, train_labels, valid_labels = get_dataset()


if __name__ == "__main__":
    main()


