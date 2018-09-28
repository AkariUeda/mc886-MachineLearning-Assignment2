
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

def softmax(x):
    s = np.exp(x)
    s = np.true_divide(s,np.sum(s))
    return s

