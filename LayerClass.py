import numpy as np
from functions import relu, reluDerivative, sigmoid, sigmoidDerivative, softmax, softmax_derivative, identidade
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, random, input_size, output_size):
        self.activation = np.array((output_size,1))
        if random == True:
            self.weights = np.random.uniform(-0.05,0.05,(input_size, output_size))
            self.bias = np.random.uniform(-0.05,0.05,(output_size,1))
        else:
            self.weights = np.identity(input_size)
            self.bias = np.zeros(output_size)

class NeuralNetwork:
    def __init__(self):
        self.camadas = []
        self.functions = []
        self.derivatives = []
        self.train_loss = []
        self.valid_loss = []

    def calc_loss(self,H,y,group):
        Y = np.zeros((len(y),10))
        for i in range(0, len(y)):
            Y[i][y[i]] = 1
        out = len(self.camadas)-1
        m = H.shape[0]
        if group == 'train':
            if self.functions[out] == sigmoid:
                self.train_loss.append(np.sum(-1* np.add(np.multiply(y,(1/H)) , np.multiply(np.subtract(1, y),(1/np.subtract(1,H))))))
            elif self.functions[out] == softmax:
                cost = np.sum(-Y * np.log(H))/m
                self.train_loss.append(cost)

        elif group == 'valid':
            if self.functions[out] == sigmoid:
                self.valid_loss.append(np.sum(-1* np.add(np.multiply(y,(1/H)) , np.multiply(np.subtract(1, y),(1/np.subtract(1,H))))))
            elif self.functions[out] == softmax:
                cost = np.sum(-Y * np.log(H))/m
                self.valid_loss.append(cost)
        return 

    def forward(self,X,y):
        out = len(self.camadas)-1
        inp = 0
        self.camadas[inp].activation = X
        for i in range(1,len(self.camadas)):
            self.camadas[i].activation = self.functions[i](np.add(self.camadas[i-1].activation.dot(self.camadas[i].weights),self.camadas[i].bias.T))
        self.calc_loss(self.camadas[out].activation, y, 'train')
        return self.camadas[out].activation

    def forward_pred(self,activation,y):
        for i in range(1,len(self.camadas)):
            activation = self.functions[i](np.add(activation.dot(self.camadas[i].weights),self.camadas[i].bias.T))
        self.calc_loss(activation, y, 'valid')
        return activation

    def backward(self,  X, y, learning_rate, lamb):
        out = len(self.camadas)-1
        m = X.shape[0]
        if self.functions[out] == softmax:
            Y = np.zeros((len(y),10))
            for i in range(0, len(y)):
                Y[i][y[i]] = 1
            self.camadas[out].delta = np.subtract(self.camadas[out].activation,Y)

        if self.functions[out] == sigmoid:
            self.camadas[out].error = self.camadas[out].activation - y
            self.camadas[out].delta = self.camadas[out].error*self.derivatives[out](self.camadas[out].activation)
            
        for i in range(len(self.camadas)-2,0,-1):
            self.camadas[i].error = self.camadas[i+1].delta.dot(self.camadas[i+1].weights.T)
            self.camadas[i].delta = self.camadas[i].error*self.derivatives[i](self.camadas[i].activation)

        for i in range(len(self.camadas)-1,0,-1):
            w = learning_rate*self.camadas[i-1].activation.T.dot(self.camadas[i].delta) + (lamb/m)*self.camadas[i].weights
            self.camadas[i].weights -= w            
            b = np.sum(self.camadas[i].delta, axis=0)
            b = b.reshape((len(b),1))
            self.camadas[i].bias -= b * learning_rate

    def predict(self,X,y):
        camadas = np.copy(self.camadas)
        out = len(camadas)-1
        camadas[0].activation = X
        for i in range(1,len(camadas)):
            camadas[i].activation = sigmoid(camadas[i-1].activation.dot(camadas[i].weights))
        preds = camadas[out].activation
        preds[preds > 0.5] = 1
        preds[preds <= 0.5] = 0
        acc = sum(preds == y)
        print("Acurácia validação: "+str(acc/len(y)))

    def train_onevsall(self,X,y,lamb,learning_rate,bs,iteracoes, printacc):
        acc = 0
        lim = int(X.shape[0]/bs)
        for i in range(0, iteracoes): 
            for j in range(0,lim):
                Xsl = X[bs*j:bs*j+bs]
                ysl = y[bs*j:bs*j+bs]
                self.forward(Xsl,ysl)
                self.backward(Xsl,ysl,learning_rate,lamb)
                preds = np.copy(self.camadas[1].activation)
                preds[preds > 0.5] = 1
                preds[preds <=0.5] = 0
                acc = 0
                for i in range(len(preds)):
                    if preds[i] == ysl[i]:
                        acc+=1
                acc = acc/len(ysl)
                if printacc:               
                    print("Acc: "+str(acc))
        return acc

    def train_neuralnet(self,X,y, Xv, yv, lamb, learning_rate,bs,iteracoes, printacc):    
        lim = int(X.shape[0]/bs)
        p_train = []
        p_valid = []
        vbs = int(Xv.shape[0]/lim)
        for i in range(0, iteracoes):
            for j in range(0,lim):
                Xsl = X[bs*j:bs*j+bs]
                ysl = y[bs*j:bs*j+bs]
                Xslv = Xv[vbs*j:vbs*j+vbs]
                yslv = yv[vbs*j:vbs*j+vbs]
                #print(X.shape,y.shape,Xv.shape,yv.shape)
                #print(Xsl.shape,ysl.shape,Xslv.shape,yslv.shape)
                pt = self.forward(Xsl,ysl)
                pv = self.forward_pred(Xslv,yslv)
                self.backward(Xsl,ysl,learning_rate, lamb)
                p_train.extend(np.argmax(pt, axis=1))
                p_valid.extend(np.argmax(pv, axis=1))
            
        yl = y.reshape((y.shape[0]))
        yvl = yv.reshape((yv.shape[0]))
        acc_train = 0
        acc_valid = 0
        for i in range(len(yl)):
            if p_train[i] == yl[i]:
                acc_train+=1
        for i in range(len(yvl)):
            if p_valid[i] == yvl[i]:
                acc_valid+=1
        #print(acc_train)
        #print(acc_valid)
        #Essa expressao chique simplismente nao funcionava pro que a gnt queria
        #acc_train = np.sum(np.array(p_train) == np.array(y1))/len(y1)
        #acc_valid = np.sum(np.array(p_valid) == np.array(yv1))/len(yv1)
        acc_train = acc_train/len(yl)
        acc_valid = acc_valid/len(yvl)
        if printacc:               
            print("Acc treino: "+str(acc_train))
            print("Acc valid: "+str(acc_valid))
            confusion_matrix = np.zeros((10,10))
            for j in range(0,10):
                for k in range(0,10):
                    confusion_matrix[j][k] += 1
            plt.plot( range(0,len(self.valid_loss)), self.valid_loss, 'g-', label='Valid')
            plt.title('title')
            plt.ylabel('Cost')
            plt.xlabel('Iterations')
            plt.legend()
            plt.savefig('training.png')
            plt.show() 
            
        return self.valid_loss



    def predict_prob(self,X,y):
        camadas = np.copy(self.camadas)
        out = len(camadas)-1
        inp = 0
        camadas[0].activation = X
        for i in range(1,len(camadas)):
            camadas[i].activation = self.functions[i](camadas[i-1].activation.dot(camadas[i].weights))
        preds = camadas[out].activation
        return preds
        
        
        
