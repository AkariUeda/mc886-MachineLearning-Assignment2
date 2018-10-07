import numpy as np
import math
import pandas as pd
from LayerClass import NeuralNetwork
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
    modelo = sys.argv[1]
    #print("Vamos fazer one vs all no toy set!")
    cl = NeuralNetwork()
    cl.load_model(modelo)
    cl.get_results(Xv,yv,"exp5-teste")
    

if __name__ == "__main__":
    main()


