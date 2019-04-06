import numpy as np
import matplotlib.pyplot as plt
import pickle

N = 10000
k = 10
eps = np.finfo(float).eps

def loadBatch(filename):
    with open("./cifar-10-batches-py/" + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = np.array(dict[b'data']/255).T
    y = np.array(dict[b'labels'])
    
    Y = np.zeros((k,N))
    for i in range(N):
        Y[y[i]][i] = 1
    
    fo.close()
    return X,Y,y

def initialize():
    W = np.random.normal(0.0, 0.01, (10,3072))
    b = np.random.normal(0.0, 0.01, (10,1))

    return W, b

def evaluateClassifier(X, W, b):
    s = np.dot(W, X) + b
    P = np.exp(s) / np.sum(np.exp(s), axis=0)
    return P

def computeCost(P, Y, W, l):
    p_y = np.multiply(Y,P).sum(axis=0)
    p_y[p_y == 0] = eps
    return -np.log(p_y).sum() / (P.shape[1] + l * np.power(W,2).sum())

def computeAccuracy(P, y):
    predictions = np.argmax(P, axis=0)
    return np.sum(predictions == y) / P.shape[1]

def computeGradients(X, Y, W, l, b):
    grad_W = np.zeros_like(W)
    grad_b = np.zeros_like(b)
    
    P = evaluateClassifier(X,W,b)
    g = -(Y-P)
    grad_b = np.dot(g,np.ones((X.shape[1],1))) / X.shape[1]
    grad_W = np.add(np.dot(g,X.T) / X.shape[1], 2*l*W)

    return grad_W, grad_b

def computeGradientsNumerically(X,Y,W,b,l):
    grad_W = np.zeros_like(W)
    grad_b = np.zeros_like(b)

    P = evaluateClassifier(X,W,b)
    c = computeCost(P,Y,W,l)
    h = 1e-6

    for i in range(b.shape[0]):
        b[i] += h
        P = evaluateClassifier(X,W,b)
        c2 = computeCost(P,Y,W,l)
        grad_b[i] = (c2 - c) / h
        b[i] -= h
        # print("b iteration " + str(i))

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i][j] += h
            P = evaluateClassifier(X,W,b)
            c2 = computeCost(P,Y,W,l)
            grad_W[i][j] = (c2 - c) / h
            W[i][j] -= h
            # print("w iteration " + str(i) + " " + str(j))

    return grad_W, grad_b

def compareGradients(X,Y,W,l,b):
    grad_W, grad_b = computeGradients(X,Y,W,l,b)
    grad_W_num, grad_b_num = computeGradientsNumerically(X,Y,W,b,l)
    print("Relative error for W: " + str(np.abs(grad_W - grad_W_num).sum()))
    print("Relative error for b: " + str(np.abs(grad_b - grad_b_num).sum()))

if __name__ == "__main__":
    X,Y,y = loadBatch("data_batch_1")
    W,b = initialize()
    l = 0.0
    # P = evaluateClassifier(X,W,b)
    # J = computeCost(P,Y,W,l)
    # acc = computeAccuracy(P,y)
    # print(computeGradients(X, Y, W, l, b))
    # print(computeGradientsNumerically(X,Y,W,b,l))
    compareGradients(X[:500,0:100], Y[:,0:100], W[:,:500],l,b)

