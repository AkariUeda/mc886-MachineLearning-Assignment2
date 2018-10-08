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

    # dataset = pd.read_csv('fashion-mnist_train.csv')
    # dataset = dataset.sample(frac=1)
    # dataset = np.array(dataset)
    # train_set    = dataset[0:50000,  2:]
    # valid_set    = dataset[50000:,2:]
    # train_labels = dataset[0:50000,1]
    # valid_labels = dataset[50000:,1]
    # np.savetxt('fashion-mnist_valid-set.csv', valid_set, delimiter=',',fmt='%i')
    # np.savetxt('fashion-mnist_train-labels.csv', train_labels, delimiter=',',fmt='%i')
    # np.savetxt('fashion-mnist_valid-labels.csv', valid_labels, delimiter=',',fmt='%i')
    # np.savetxt('fashion-mnist_train-set.csv', train_set, delimiter=',',fmt='%i')

    # dataset = pd.read_csv('toy_fashion-mnist_train.csv')

    # dataset = dataset.sample(frac=1)
    # dataset = np.array(dataset)
    # train_set    = dataset[0:400,  2:]
    # valid_set    = dataset[400:,2:]
    # train_labels = dataset[0:400,1]
    # valid_labels = dataset[400:,1]

    print("Lendo dataset...")


    train_set    = np.genfromtxt('fashion-mnist_train-set.csv', delimiter=',')[1:,1:]
    valid_set    = np.genfromtxt('fashion-mnist_valid-set.csv', delimiter=',')[1:,1:]
    train_labels = np.genfromtxt('fashion-mnist_train-labels.csv', delimiter=',')[:,1]
    valid_labels = np.genfromtxt('fashion-mnist_valid-labels.csv', delimiter=',')[:,1]

    # np.savetxt('toy_fashion-mnist_valid-set.csv', valid_set, delimiter=',')
    # np.savetxt('toy_fashion-mnist_train-labels.csv', train_labels, delimiter=',')
    # np.savetxt('toy_fashion-mnist_valid-labels.csv', valid_labels, delimiter=',')
    # np.savetxt('toy_fashion-mnist_train-set.csv', train_set, delimiter=',')
    # train_set    = np.genfromtxt('toy_fashion-mnist_train-set.csv', delimiter=',')
    # valid_set    = np.genfromtxt('toy_fashion-mnist_valid-set.csv', delimiter=',')
    # train_labels = np.genfromtxt('toy_fashion-mnist_train-labels.csv', delimiter=',')
    # valid_labels = np.genfromtxt('toy_fashion-mnist_valid-labels.csv', delimiter=',')

    #print(train_set.shape, valid_set.shape, train_labels.shape, valid_labels.shape)


    print("Dataset carregado com sucesso")

    train_set = train_set.astype(int)
    valid_set = valid_set.astype(int)
    train_labels = train_labels.astype(int)
    valid_labels = valid_labels.astype(int)

    mean = np.mean(train_set, axis=0)
    std = np.std(train_set, axis=0)
    std[std == 0] = 1
    train_set = normalize_features(train_set, mean, std)
    valid_set = normalize_features(valid_set, mean, std)
    train_labels = train_labels.reshape((len(train_labels),1))
    valid_labels = valid_labels.reshape((len(valid_labels),1))

    print("Dataset normalizado")
    return train_set, valid_set, train_labels, valid_labels

if __name__ == "__main__":
    main()
