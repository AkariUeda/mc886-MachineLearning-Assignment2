import numpy as np
import pandas as pd

def normalize_features(features, mean, std):
    output = np.copy(features)
    output = np.subtract(output, mean)
    output = np.divide(output, std)
    return output

def main():
    ##
    ##          Lendo e preparando o Dataset
    ##

    dataset = pd.read_csv('fashion-mnist_test.csv')   
    dataset = np.array(dataset)
    test_set  = dataset[:,1:]
    test_labels = dataset[:,0]
    np.savetxt('fashion-mnist_test-set.csv', test_set, delimiter=',',fmt='%i')
    np.savetxt('fashion-mnist_test-labels.csv', test_labels, delimiter=',',fmt='%i')    

    print("Lendo dataset...")

    train_set    = np.genfromtxt('fashion-mnist_train-set.csv', delimiter=',')[1:,1:]
    test_set    = np.genfromtxt('fashion-mnist_test-set.csv', delimiter=',')
    train_labels = np.genfromtxt('fashion-mnist_train-labels.csv', delimiter=',')[:,1]
    test_labels = np.genfromtxt('fashion-mnist_test-labels.csv', delimiter=',')

    print("Dataset carregado com sucesso")

    train_set = train_set.astype(int)
    test_set = test_set.astype(int)
    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)

    mean = np.mean(train_set, axis=0)
    std = np.std(train_set, axis=0)
    std[std == 0] = 1
    train_set = normalize_features(train_set, mean, std)
    test_set = normalize_features(test_set, mean, std)
    train_labels = train_labels.reshape((len(train_labels),1))
    test_labels = test_labels.reshape((len(test_labels),1))

    print("Dataset normalizado")
    return test_set, test_labels

if __name__ == "__main__":
    main()
