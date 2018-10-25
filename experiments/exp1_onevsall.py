import numpy as np
import math
import pandas as pd
from LayerClass import OneVsAllClassifier
from random import randrange, uniform
from functions import *
import get_dataset
import sys

##
## ONE VS ALL
##


def main():
    train_set, valid_set, train_labels, valid_labels = get_dataset.main()
    X = train_set
    y = train_labels
    Xv = valid_set
    yv = valid_labels
    iteracoes_grid = int(sys.argv[1])
    iteracoes_train = int(sys.argv[2])
    batch_size = 256
    print_acc = True
    alpha = 0.02
    lamb = 0.001
    alpha, lamb = grid_search(OneVsAllClassifier, X, y, iteracoes_grid)
    #print("Vamos fazer one vs all no toy set!")
    cl = OneVsAllClassifier(X)
    cl.train_neuralnet(X,y,Xv,yv,alpha,lamb,batch_size,iteracoes_train,print_acc, 'oneVall')
   

if __name__ == "__main__":
    main()


