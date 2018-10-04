import numpy as np
import math
import matplotlib.pyplot as plt
import random

def identidade(x):
    return x

def relu(x):
    return np.maximum(x,0)
    
def reluDerivative(x):
    return np.greater(x, 0).astype(int)

def sig(x):
  x = np.clip( x, -500, 500 )
  x = 1 / (1 + math.exp(-x))

  return x

def sigmoid(v):
    return np.vectorize(sig)(v)

def sigmoidDerivative(x):
    return x * (1 - x)

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)
    
def softmax_derivative(X):
    res = []
    exps = np.sum(np.exp(X))
    for i in range(0, len(X)):
        r = X[i]*(exps - np.exp(X[i])) / np.power(exps,2)
        res.append(r)
    res = np.array(res)
    #print(res.shape)
    return res

def cross_entropy(X,y):
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss

def delta_cross_entropy(X,y):
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad

def grid_search(model, X,y, iteracoes):
        learning_rates = [2e-3, 2e-4, 2e-5 ,2e-6,2e-7]
        lambdas = [1e-1, 1e-2, 1e-3, 1e-4]
        media_minloss = 10e10
        batch_size = 32
        # Montando sub-conjunto de validação
        m = X.shape[0]
        n = math.floor(0.7 * m)
        
        print("Running grid search...")
        print("Learning rate    Lambda       Loss")
        for alpha in learning_rates:
            for lamb in lambdas: 
                    loss_fold = 0
                    for fold in range(0,5):
                        index = random.sample(range(m),m)
                        Xv = X[index[n:]]
                        yv = y[index[n:]]
                        Xt = X[index[:n]]
                        yt = y[index[:n]]
                        test_model = model(Xt)
                        loss = test_model.train_neuralnet(Xt, yt, Xv, yv, lamb, alpha,batch_size,  iteracoes, False)
                        loss_fold += loss[-1]
                    loss_fold /= 5
                    print("  {:<15}  {:<10} {:<5.4}".format(alpha, lamb, loss_fold))
                    if loss_fold < media_minloss:
                        media_minloss = loss_fold
                        best_lr = alpha
                        best_lamb = lamb

        print("Best learning rate: "+str(best_lr))
        print("Best regularization lambda: "+str(best_lamb))



        return best_lr, best_lamb
