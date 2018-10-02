import numpy as np

def identidade(x):
    return x

def relu(x):
    return np.max(x,0)

def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sig(x):
  x = np.clip( x, -500, 500 )
  x = 1 / (1 + math.exp(-x))
  if x > 0.999:
    return 0.999
  if x < 0.0001:
    return 0.0001
  return x

sigmoid = np.vectorize(sig)

def sigmoidDerivative(x):
    return x * (1 - x)


def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


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
