import numpy as np

class Layer:
    def __init__(self, random, input_size, output_size):
        if random == True:
            print("pesos aleatorios")
            self.weights = np.random.uniform(-0.05,0.05,(input_size, output_size))
        else:
            self.weights = np.identity(input_size)

    def loss(self,Y):
        H = self.activation
        return np.mean(np.subtract(np.multiply(np.multiply(-1, Y),np.log(H)),np.multiply(np.subtract(1, Y),np.log(np.subtract(1,H)))))

    def back(camadas, X, y, learning_rate):
        out = len(camadas)-1
        inp = 0
        camadas[out].error = np.subtract(y,camadas[out].activation)
        camadas[out].delta = camadas[out].error*sigmoidDerivative(camadas[out].activation)

        for i in range(len(camadas)-2,0,-1):
            print(i)
            camadas[i].error = camadas[i+1].delta.dot(camadas[i].weights)
            camadas[i].delta = camadas[i].error*sigmoidDerivative(camadas[i].activation)

        for i in range(len(camadas)-1,0,-1):
            camadas[i].weights += learning_rate*camadas[i-1].activation.T.dot(camadas[i].delta)

class NeuralNetwork:
    def __init__(self, random, input_size, output_size):
        self.camadas = []

    def forward(self,X,y):
        out = len(self.camadas)-1
        inp = 0
        self.camadas[0].activation = X
        for i in range(1,len(self.camadas)):
            self.camadas[i].activation = sigmoid(self.camadas[i-1].activation.dot(self.camadas[i].weights))
        return self.camadas[out].activation

    def predict(self,X,y):
        camadas = np.copy(self.camadas)
        out = len(camadas)-1
        inp = 0
        camadas[0].activation = X
        for i in range(1,len(camadas)):
            camadas[i].activation = sigmoid(camadas[i-1].activation.dot(camadas[i].weights))

        output = camadas[out].activation
        preds = camadas[out].activation
        preds[preds > 0.5] = 1
        preds[preds <=0.5] = 0
        acc = sum(preds == y)
        print("Acurácia validação: "+str(acc/len(y)))

    def train(self,X,y,learning_rate,iteracoes, printacc):
        for i in range(0, iteracoes):
            self.forward(self.camadas,X,y)
            self.back(self.camadas,X,y,learning_rate)
            if printacc:
                preds = np.copy(self.camadas[1].activation)
                preds[preds > 0.5] = 1
                preds[preds <=0.5] = 0
                acc = sum(preds == y)
                print("Acc: "+str(acc/len(y)))

    def predict_prob(self,X,y):
        camadas = np.copy(self.camadas)
        out = len(camadas)-1
        inp = 0
        camadas[0].activation = X
        for i in range(1,len(camadas)):
            camadas[i].activation = sigmoid(camadas[i-1].activation.dot(camadas[i].weights))
        output = camadas[out].activation
        preds = camadas[out].activation
        return preds
