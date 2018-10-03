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
        
    def train(self,X,yl,lr,lb,it):        
        self.neural_net = []
        for i in range(self.classes):
            self.neural_net.append(NeuralNetwork())
        #y = np.zeros((yl.shape[0],self.classes))
        y = np.repeat(yl,10,axis=1)
        y = y.T
      
        for i in range(self.classes):
            nc = y[i].reshape((400, 1))
            print("Vamos marcar todos os "+str(i)+" !!")
            #print(nc.T)
            for j in range(len(nc)):
                nc[j] = nc[j] == i 
            #nc[nc == i] = 1
            #nc[nc != i] = 0
            #print(nc.T)
            # Adicionando a camada de input no indice 0
            
            self.neural_net[i].camadas.append(Layer(False, X.shape[1], X.shape[1]))
            self.neural_net[i].functions.append(identidade) 
            self.neural_net[i].derivatives.append(identidade)
            self.neural_net[i].forward(X,y)
            #print("Primeira camada adicionada")

            # Adicionando a camada de saida no indice 1
            self.neural_net[i].camadas.append(Layer(True,self.neural_net[i].camadas[0].activation.shape[1], 1 ))
            self.neural_net[i].functions.append(sigmoid)
            self.neural_net[i].derivatives.append(sigmoidDerivative)
            #print("Segunda camada adicionada") 
            acc = self.neural_net[i].train_onevsall(X,nc,lr,lb,it,False)
            print("Acc de "+str(acc)+"% para a classe "+str(i))

    def classify(self,X,y):
        probs = [];
        for i in range(self.classes):
            probs.append(self.neural_net[i].predict_prob(X,y))
        return np.argmax(probs,axis=0)
    
    def classDumb(self,X,y):
        res = []
        for i in range(len(X)):
            res.append(self.classSingle(X[i],y[i],i))
        return res
      
    def classSingle(self,x,y,id):
        bstp = -1
        cl = -1
        debug = []
        for i in range(self.classes):
            nval = self.neural_net[i].predict_prob(x,y)
            if nval > bstp:
                bstp = nval
                cl = i
            debug.append("O item "+str(id)+" pertence a "+str(i)+" com chance "+str(nval)+" deveria ser "+str(y))
        ## UMA CLASSE SO PRA TESTE ##
        #if bstp < 0.50:
        #    cl = 1
        #else:
        #    cl = 0        
        #############################
        if(cl != y):
            for l in debug:
                print(l)
        return cl
def main():
    train_set, valid_set, train_labels, valid_labels = get_dataset.main()
    X = train_set
    y = train_labels
    Xv = valid_set
    yv = valid_labels
    ## UMA CLASSE SO PRA TESTE ##
    #for j in range(len(y)):
    #            y[j] = not(y[j] == 0) 
    #            
    #for j in range(len(yv)):
    #            yv[j] = not(yv[j] == 0)
    #############################
    print("Vamos fazer one vs all no toy set!")
    cl = OneVsAllClassifier(10)
    cl.train(X,y,0.002,0.02,100)
    results = cl.classify(Xv,yv)
    erro = 0    
    #erro = sum(results != yv)
    for i in range(len(results)):
        erro += results[i] != yv[i]
        print(str(i)+" foi classificado: "+str(results[i])+" VS esperado "+str(train_labels[i]))
    print("Erramos "+str(erro)+" previsoes no chique")

if __name__ == "__main__":
    main()

