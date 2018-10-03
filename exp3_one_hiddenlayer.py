import numpy as nppr
import math
import pandas as pd
from LayerClass import Layer, NeuralNetwork
from random import randrange, uniform
from functions import *
import get_dataset


##
## REDE NEURAL COM UMA CAMADA ESCONDIDA
##

def main():
    train_set, valid_set, train_labels, valid_labels = get_dataset.main()
    neural_net = NeuralNetwork()

    # Adicionando a camada de input no camada 0
    neural_net.camadas.append(Layer(False, train_set.shape[1], train_set.shape[1]))
    neural_net.functions.append(identidade)
    neural_net.derivatives.append(identidade)

    # Adicionando a camada escondida com 342 neurônios na camada 1
    neural_net.camadas.append(Layer(True, train_set.shape[1], 342))
    neural_net.functions.append(relu)
    neural_net.derivatives.append(reluDerivative)

    # Adicionando a camada de saída com 10 neurônios na camada 2
    neural_net.camadas.append(Layer(True, 342, 10))
    neural_net.functions.append(softmax)
    neural_net.derivatives.append(softmax_derivative)
    
    print(neural_net.camadas)
    print(neural_net.functions)
    print(neural_net.derivatives)
    # Treinando
    learning_rate = 0.0000002
    iteracoes = 100000
    print_acuracia = True
    neural_net.train_neuralnet(train_set, train_labels, valid_set, valid_labels, learning_rate,iteracoes, print_acuracia)

if __name__ == "__main__":
    main()
