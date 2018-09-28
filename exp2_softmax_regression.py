import numpy as np
import math
import pandas as pd
from LayerClass import Layer, NeuralNetwork
from random import randrange, uniform
from functions import *
import get_dataset


##
## REGRESSAO LOGISTICA COM REDE NEURAL DE UMA CAMADA: ONE VS. ALL
##    

def onevsall(classe, train_set, train_labels, valid_set, valid_labels):

    neural_net_logistic = []
    y_train = np.zeros((train_labels.shape))
    y_train[train_labels == classe] = 1
    y_train[train_labels != classe] = 0
    y_valid = np.zeros((valid_labels.shape))
    y_valid [valid_labels == classe] = 1
    y_valid [valid_labels != classe] = 0

    # Adicionando a camada de input no indice 0
    neural_net_logistic.append(Layer(False, train_set.shape[1], train_set.shape[1]))
    neural_net_logistic[0].forward(train_set,identidade)

    # Adicionando a camada de saída no índice 1
    neural_net_logistic.append(Layer(True, neural_net_logistic[0].activation.shape[1], 1 ))

    train(neural_net_logistic, train_set, y_train, 0.02, 1000)
    predict(neural_net_logistic, valid_set, y_valid)
    
def main():
    print("oi")


if __name__ == "__main__":
    main()
