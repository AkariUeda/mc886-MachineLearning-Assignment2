import numpy as np
import math
from functions import relu, reluDerivative, sigmoid, sigmoidDerivative, softmax, softmax_derivative, identidade
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

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
        self.best_model = []
        self.best_cost = -1

    def load_model(self,nn):
        self.camadas = []
        self.functions = []
        self.derivatives = []
        self.train_loss = []
        self.valid_loss = []
        network = np.load(nn)
        print(network.shape)
        weights = network[0]
        functions = network[1]
        derivatives = network[2]
        for i in range(weights.shape[0]):
            self.camadas.append(Layer(False,weights[i].shape[0],weights[i].shape[1]))
            self.camadas[i].weights = weights[i].copy()
            self.functions.append(functions[i])
            self.derivatives.append(derivatives[i])

    def model_to_list(self):
        ws = [[],[],[]]
        for i in range(len(self.camadas)):            
            ws[0].append(self.camadas[i].weights)
            ws[1].append(self.functions[i])
            ws[2].append(self.derivatives[i])
        self.best_model = ws

    def save_model(self,name):        
        ws = np.array(self.best_model)
        np.save(name, ws)

    def calc_loss(self,H,y,group):
        Y = np.zeros((len(y),10))
        for i in range(0, len(y)):
            Y[i][y[i]] = 1
        out = len(self.camadas)-1
        m = H.shape[0]
        cost = 0
        if self.functions[out] == sigmoid:
            cost = np.sum(-1* np.add(np.multiply(y,(1/H)) , np.multiply(np.subtract(1, y),(1/np.subtract(1,H)))))
        elif self.functions[out] == softmax:
            cost = np.sum(-1*np.sum(Y *np.log(H+1e-9))/m)
        if group == 'train':        
            if self.functions[out] == sigmoid:
                self.train_loss.append(cost)
            elif self.functions[out] == softmax:
                self.train_loss.append(cost)
        elif group == 'valid':
            if self.functions[out] == sigmoid:
                self.valid_loss.append(cost)
            elif self.functions[out] == softmax:
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
        
    def get_results(self,X,y,experiment):
        p_valid = np.argmax(self.forward_pred(X,y),axis=1)
        y = y.reshape((y.shape[0]))
        self.print_results(experiment,y,p_valid)
        return
        
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

    def train_neuralnet(self,X,y, Xv, yv, lamb, learning_rate,bs,iteracoes, printacc, experiment):    
        lim = int(math.ceil(X.shape[0]/bs))
        vbs = int(math.ceil(Xv.shape[0]/lim))
        for i in range(0, iteracoes):
            p_train = []
            p_valid = []
            for j in range(0,lim):
                Xsl = X[bs*j:bs*j+bs]
                ysl = y[bs*j:bs*j+bs]
                pt = self.forward(Xsl,ysl)
                self.backward(Xsl,ysl,learning_rate, lamb)
                #print(np.argmax(pt, axis=1).shape, pt.shape)
                p_train.extend(np.argmax(pt, axis=1))            
            pv = self.forward_pred(Xv,yv)    
            p_valid.extend(np.argmax(pv, axis=1))
            if self.best_cost == -1 or self.best_cost > self.valid_loss[i]:
                self.best_cost = self.valid_loss[i]
                self.best_model = self.model_to_list()
        if(len(self.train_loss) != iteracoes and iteracoes != 1):
            self.train_loss = np.mean(np.array(self.train_loss).reshape(-1,lim),axis=1).tolist()

        #print(len(p_train), len(p_valid))   
        yl = y.reshape((y.shape[0]))

        yvl = yv.reshape((yv.shape[0]))
        #print(len(yl), len(yvl))
        
        if printacc:     
            self.print_results(experiment,yvl,p_valid,yl,p_train,True)

        return self.valid_loss

    def print_results(self,experiment,yvl,p_valid,yl=None,p_train=None,print_iter=False):
        acc_train = 0
        acc_valid = 0
        if(p_train != None):
            for i in range(len(yl)):
                if p_train[i] == yl[i]:
                    acc_train+=1
            acc_train = acc_train/len(yl)  
            print("Acc treino: "+str(acc_train))  
            
        for i in range(len(yvl)):
            if p_valid[i] == yvl[i]:
                acc_valid+=1
        acc_valid = acc_valid/len(yvl)
        
        confusion_matrix = np.zeros((10,10))
        for j in range(0,len(p_valid)):
            confusion_matrix[yvl[j]][p_valid[j]] += 1  
            
        
        print("Acc valid: "+str(acc_valid))
        print("Loss: "+str(self.valid_loss[-1]))

        df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "0123456789"], columns = [i for i in "0123456789"])
        plt.figure(figsize = (10,7))
        ax = sn.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
        ax.set(xlabel='Predicted', ylabel='Real')
        plt.savefig(experiment+'_confusion_matrix.png')
        plt.show()
        plt.close()
        if print_iter:
            plt.figure(1)       
            plt.subplot(211)
            plt.plot( range(0,len(self.train_loss)), self.train_loss, 'r-', label='Train')
            plt.ylabel('Cost')
            plt.subplot(212)
            plt.plot( range(0,len(self.valid_loss)), self.valid_loss, 'g-', label='Valid')
            plt.ylabel('Cost')
            plt.xlabel('Iterations')
            plt.legend()
            plt.savefig(experiment+'_training.png')
            plt.show()
        return 

    def predict_prob(self,X,y):
        camadas = np.copy(self.camadas)
        out = len(camadas)-1
        inp = 0
        camadas[0].activation = X
        for i in range(1,len(camadas)):
            camadas[i].activation = self.functions[i](camadas[i-1].activation.dot(camadas[i].weights))
        preds = camadas[out].activation
        return preds
        
class OneVsAllClassifier:
    def __init__(self,X):
        self.classes = 10
        self.valid_loss = [] 
        self.train_loss = []   
        self.neural_net = []
        self.best_model = []
        self.best_cost = -1
        for i in range(self.classes):
            self.neural_net.append(NeuralNetwork())
        for i in range(self.classes):
            self.neural_net[i].camadas.append(Layer(False, X.shape[1], X.shape[1]))
            self.neural_net[i].functions.append(identidade) 
            self.neural_net[i].derivatives.append(identidade)
            self.neural_net[i].camadas.append(Layer(True,X.shape[1],1))
            self.neural_net[i].functions.append(sigmoid)
            self.neural_net[i].derivatives.append(sigmoidDerivative)

    def calc_loss(self,H,y,Hv,yv):
        Y = np.zeros((len(y),10))
        for i in range(0, len(y)):
            Y[i][y[i]] = 1      
        Yv = np.zeros((len(yv),10))
        for i in range(0, len(yv)):
            Yv[i][yv[i]] = 1     
        m = H.shape[0]
        mv = Hv.shape[0]
        vl = 0
        tl = 0
        
        self.train_loss.append(-1*np.sum(Y *np.log(H+1e-9))/m)
        self.valid_loss.append(-1*np.sum(Yv * np.log(Hv+1e-9))/mv)
        return 

    def train_neuralnet(self,X,yl,Xv,yvl,lr,lb,bs,it,printacc,experiment):    
        print("Training model...")
        #y = np.zeros((yl.shape[0],self.classes))
        y = np.repeat(yl,self.classes,axis=1).T
        yv = np.repeat(yvl,self.classes,axis=1).T
         
        nc = []
        ncv = []
        for i in range(self.classes):            
            nc.append(y[i].reshape((y[i].shape[0], 1)))
            ncv.append(yv[i].reshape((yv[i].shape[0], 1)))
            for j in range(len(nc[i])):
                nc[i][j] = nc[i][j] == i 
            for j in range(len(ncv[i])):
                ncv[i][j] = ncv[i][j] == i 

        for i in range(it):    
            for j in range(self.classes):   
                self.neural_net[j].train_neuralnet(X,nc[j],Xv,ncv[j],lr,lb,bs,1,False,experiment)            
            r = self.classify(X,y,raw=True)
            rs = np.zeros([r.shape[0],r.shape[1]])
            for k in range(rs.shape[0]):
                 for l in range(rs.shape[1]):
                    rs[k][l] = np.asscalar(r[k][l])
            results = rs.T
            
            r = self.classify(Xv,yv,raw=True)
            rs = np.zeros([r.shape[0],r.shape[1]])
            for k in range(rs.shape[0]):
                 for l in range(rs.shape[1]):
                    rs[k][l] = np.asscalar(r[k][l])
            resultsv = rs.T
            self.calc_loss(results,yl,resultsv,yvl) 
           
        p_train = self.classify(X,y)
        p_valid = self.classify(Xv,yv)

        acc_train = 0
        acc_valid = 0
        for i in range(len(yl)):
            if p_train[i] == yl[i]:
                acc_train+=1
        for i in range(len(yvl)):
            if p_valid[i] == yvl[i]:
                acc_valid+=1
        acc_train = acc_train/len(yl)  
        acc_valid = acc_valid/len(yvl)

        if printacc:     
            self.print_results(experiment,yvl,p_valid,yl,p_train.tolist(),True)

        return self.valid_loss

    def print_results(self,experiment,yvl,p_valid,yl=None,p_train=None,print_iter=False):
        acc_train = 0
        acc_valid = 0
        if(p_train != None):
            for i in range(len(yl)):
                if p_train[i] == yl[i]:
                    acc_train+=1
            acc_train = acc_train/len(yl)  
            print("Acc treino: "+str(acc_train))  
            
        for i in range(len(yvl)):
            if p_valid[i] == yvl[i]:
                acc_valid+=1
        acc_valid = acc_valid/len(yvl)
        
        confusion_matrix = np.zeros((10,10))
        for j in range(0,len(p_valid)):
            confusion_matrix[yvl[j][0]][p_valid[j]] += 1  
            
        
        print("Acc valid: "+str(acc_valid))
        print("Loss: "+str(self.valid_loss[-1]))

        df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "0123456789"], columns = [i for i in "0123456789"])
        plt.figure(figsize = (10,7))
        ax = sn.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
        ax.set(xlabel='Predicted', ylabel='Real')
        plt.savefig(experiment+'_confusion_matrix.png')
        plt.show()
        plt.close()
        if print_iter:
            plt.figure(1)       
            plt.subplot(211)
            plt.plot( range(0,len(self.train_loss)), self.train_loss, 'r-', label='Train')
            plt.ylabel('Cost')
            plt.subplot(212)
            plt.plot( range(0,len(self.valid_loss)), self.valid_loss, 'g-', label='Valid')
            plt.ylabel('Cost')
            plt.xlabel('Iterations')
            plt.legend()
            plt.savefig(experiment+'_training.png')
            plt.show()
        return 
    
    
    def classify(self,X,y,raw=False):
        probs = [];
        for i in range(self.classes):
            probs.append(self.neural_net[i].predict_prob(X,y))
        preds = np.argmax(probs,axis=0)  
        res = np.zeros(len(preds))      
        for i in range(len(preds)):
            res[i] = int(preds[i][0])
        res = res.astype(int)

        if raw:
            return np.array(probs)
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
        if(cl != y):
            for l in debug:
                print(l)
        return cl        
        
