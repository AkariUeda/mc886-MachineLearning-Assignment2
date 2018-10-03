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

def grid_search(model, X,y):
        learning_rates = [2e-3, 2e-4, 2e-5 ,2e-6,2e-7]
        iteracoes = 10
        lambdas = [0.75, 1, 0.5, 0.25]
        min_loss = [10e10]

        # Montando sub-conjunto de validação
        m = X.shape[0]
        n = math.floor(0.7 * m)

        
        print("Running grid search...")

        for alpha in learning_rates:
            for lamb in lambdas:
                    index = random.sample(range(m),m)
                    Xv = X[index[n:]]
                    yv = y[index[n:]]
                    Xt = X[index[:n]]
                    yt = y[index[:n]]
                    test_model = model(Xt)
                    loss = test_model.train_neuralnet(Xt, yt, Xv, yv, lamb, alpha, iteracoes, False)
                    if loss[-1] < min_loss[-1]:
                        min_loss = np.copy(loss)
                        best_lr = alpha
                        best_lamb = lamb

        print("Best learning rate: "+str(best_lr))
        print("Best regularization lambda: "+str(best_lamb))

        plt.plot( range(0,iteracoes), min_loss, 'g-', label='Valid')
        plt.title('title')
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.legend()
        plt.savefig('grid_search.png')
        plt.show() 