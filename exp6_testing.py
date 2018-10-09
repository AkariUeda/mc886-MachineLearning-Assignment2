import numpy as np
import math
import pandas as pd
from LayerClass import NeuralNetwork
from random import randrange, uniform
from functions import *
import get_testset
import sys

##
## ONE VS ALL
##

def main():
    modelo = sys.argv[1]
    print(modelo)
    test_set, test_labels = get_testset.main()
    X = test_set
    y = test_labels
    
    #print("Vamos fazer one vs all no toy set!")
    cl = NeuralNetwork()
    cl.load_model(modelo)
    cl.get_results(X,y,"exp5-teste")

if __name__ == "__main__":
    main()


