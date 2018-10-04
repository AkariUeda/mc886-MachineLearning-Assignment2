import numpy as nppr
import math
import pandas as pd
from LayerClass import Layer, NeuralNetwork
from random import randrange, uniform
from functions import *
import get_dataset
import sys


##
## REGRESSAO SOFTMAX COM REDE NEURAL DE UMA CAMADA
##


def new_neuralnet(train_set):
    neural_softmax = NeuralNetwork()

    # Adicionando a camada de input no indice 0
    neural_softmax.camadas.append(Layer(False, train_set.shape[1], train_set.shape[1]))
    neural_softmax.functions.append(identidade)
    neural_softmax.derivatives.append(identidade)

    # Adicionando a camada de saída com 10 neurônios no índice 1
    neural_softmax.camadas.append(Layer(True, train_set.shape[1], 10))
    neural_softmax.functions.append(softmax)
    neural_softmax.derivatives.append(softmax_derivative)

    return neural_softmax

def main():
    train_set, valid_set, train_labels, valid_labels = get_dataset.main()

    neural_softmax = new_neuralnet(train_set)

    iteracoes_grid = int(sys.argv[1])
    iteracoes_train = int(sys.argv[2])

    # Treinando
    batch_size = 256
    learning_rate, lamb = grid_search(new_neuralnet, train_set, train_labels, iteracoes_grid)
    print_acuracia = True
    neural_softmax.train_neuralnet(train_set, train_labels, valid_set, valid_labels, lamb, learning_rate,batch_size,iteracoes_train, print_acuracia)

if __name__ == "__main__":
    main()
