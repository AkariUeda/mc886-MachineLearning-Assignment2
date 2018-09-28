import numpy as np
import math
import pandas as pd
from LayerClass import Layer, NeuralNetwork
from random import randrange, uniform
from functions import *
import get_dataset


##
## REGRESSAO SOFTMAX COM REDE NEURAL DE UMA CAMADA
##

def main():
    train_set, valid_set, train_labels, valid_labels = get_dataset.main()

    neural_softmax = NeuralNetwork()

    # Adicionando a camada de input no indice 0
    neural_softmax.camadas.append(Layer(False, train_set.shape[1], train_set.shape[1]))
    neural_softmax.camadas[0].forward(train_set,identidade)

    # Adicionando a camada de saída com 10 neurônios no índice 1
    neural_softmax.camadas.append(Layer(True, neural_softmax.camadas[0].activation.shape[1], 10))

    # Treinando
    learning_rate = 0.02
    iteracoes = 1000
    print_acuracia = True

    neural_softmax.train(train_set, train_labels,learning_rate,iteracoes, print_acuracia)


if __name__ == "__main__":
    main()
