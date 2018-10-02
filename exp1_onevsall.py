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
        self.neural_net = [NeuralNetwork() for i in range(self.classes)]
        #y = np.zeros((yl.shape[0],self.classes))
        y = np.repeat(yl,10,axis=1)
        for i in range(self.classes):
            y[i][y[i] == i] = 1
            y[i][y[i] != i] = 0
            # Adicionando a camada de input no indice 0
            print("Forma do X")
            print(i)
            self.neural_net[i].camadas.append(Layer(False, X.shape[1], X.shape[1]))
            self.neural_net[i].forward(X,identidade)
            #print("Primeira camada adicionada")

            # Adicionando a camada de saida no indice 1
            self.neural_net[i].camadas.append(Layer(True,self.neural_net[i][0].activation.shape[1], 1 ))
            #print("Segunda camada adicionada") 
            self.neural_net[i].train(X,y[i],lr,it,False)

    def classify(X,y):
        probs = [];
        for i in range(self.classes):
            probs.append(self.neural_net.predict_proba(X,y))
        print(probs)
            

def main():
    train_set, valid_set, train_labels, valid_labels = get_dataset.main()
    cl = OneVsAllClassifier(10)
    cl.train(train_set,train_labels,0.002,1000)
    cl.classify(valid_set,valid_labels)
    
if __name__ == "__main__":
    main()


