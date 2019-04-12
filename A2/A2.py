import numpy as np
import matplotlib.pyplot as plt
import pickle

N = 10000
d = 3072
k = 10
m = 50
eps = np.finfo(float).eps

def loadBatch(filename):
    with open("./cifar-10-batches-py/" + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = np.array(dict[b'data']/255).T
    X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

    y = np.array(dict[b'labels'])
    
    Y = np.zeros((k,N))
    for i in range(N):
        Y[y[i]][i] = 1
    
    fo.close()
    return X,Y,y

def initialize():
    W1 = np.random.normal(0.0, 1/np.sqrt(d), (m,d))
    W2 = np.random.normal(0.0, 1/np.sqrt(m), (k,m))
    b1 = np.zeros((m,1))
    b2 = np.zeros((k,1))

    return W1, W2, b1, b2

def evaluateClassifier(X, W1, W2, b1, b2):
    s1 = np.dot(W1, X) + b1
    h = np.maximum(0,s1)
    s = np.dot(W2, h) + b2
    P = np.exp(s) / np.sum(np.exp(s), axis=0)
    return P,h

def computeCost(P, Y, W1, W2, l):
    p_y = np.multiply(Y,P).sum(axis=0)
    p_y[p_y == 0] = eps

    loss = -np.log(p_y).sum() / P.shape[1]
    cost = loss + l * (np.power(W1,2).sum() + np.power(W2,2).sum())
    return loss, cost

def computeAccuracy(P, y):
    predictions = np.argmax(P, axis=0)
    return np.sum(predictions == y) / P.shape[1]

def computeGradients(X, Y, W1, W2, b1, b2, l):
    grad_W1 = np.zeros_like(W1)
    grad_W2 = np.zeros_like(W2)
    grad_b1 = np.zeros_like(b1)
    grad_b2 = np.zeros_like(b2)
    
    P,h = evaluateClassifier(X,W1,W2,b1,b2)
    g = -(Y-P)
    grad_b2 = np.dot(g,np.ones((X.shape[1],1))) / X.shape[1]
    grad_W2 = np.add(np.dot(g,h.T) / X.shape[1], 2*l*W2)

    g = np.multiply(np.dot(W2.T,g),1*(h>0))
    grad_b1 = np.dot(g,np.ones((X.shape[1],1))) / X.shape[1]
    grad_W1 = np.add(np.dot(g,X.T) / X.shape[1], 2*l*W1)

    return grad_W1, grad_W2, grad_b1, grad_b2

def computeGradientsNumerically(X,Y,W1,W2,b1,b2,l):
    grad_W1 = np.zeros_like(W1)
    grad_W2 = np.zeros_like(W2)
    grad_b1 = np.zeros_like(b1)
    grad_b2 = np.zeros_like(b2)

    P,_ = evaluateClassifier(X,W1,W2,b1,b2)
    _,cost = computeCost(P,Y,W1,W2,l)
    h = 1e-5

    for i in range(b1.shape[0]):
        b1[i] += h
        P,_ = evaluateClassifier(X,W1,W2,b1,b2)
        _,c2 = computeCost(P,Y,W1,W2,l)
        grad_b1[i] = (c2 - cost) / h
        b1[i] -= h
    
    for i in range(b2.shape[0]):
        b2[i] += h
        P,_ = evaluateClassifier(X,W1,W2,b1,b2)
        _,c2 = computeCost(P,Y,W1,W2,l)
        grad_b2[i] = (c2 - cost) / h
        b2[i] -= h

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1[i][j] += h
            P,_ = evaluateClassifier(X,W1,W2,b1,b2)
            _,c2 = computeCost(P,Y,W1,W2,l)
            grad_W1[i][j] = (c2 - cost) / h
            W1[i][j] -= h
    
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2[i][j] += h
            P,_ = evaluateClassifier(X,W1,W2,b1,b2)
            _,c2 = computeCost(P,Y,W1,W2,l)
            grad_W2[i][j] = (c2 - cost) / h
            W2[i][j] -= h

    return grad_W1, grad_W2, grad_b1, grad_b2

def compareGradients(X,Y,W1,W2,b1,b2,l):
    grad_W1, grad_W2, grad_b1, grad_b2 = computeGradients(X,Y,W1,W2,b1,b2,l)
    grad_W1_num, grad_W2_num, grad_b1_num, grad_b2_num = computeGradientsNumerically(X,Y,W1,W2,b1,b2,l)
    print("Relative error for W1: " + str(np.abs(grad_W1 - grad_W1_num).sum()))
    print("Relative error for W2: " + str(np.abs(grad_W2 - grad_W2_num).sum()))
    print("Relative error for b1: " + str(np.abs(grad_b1 - grad_b1_num).sum()))
    print("Relative error for b2: " + str(np.abs(grad_b2 - grad_b2_num).sum()))

def miniBatch(W, b, l, n_epochs, n_batch, eta):
    X, Y, y = loadBatch("data_batch_1")
    X_val, Y_val, y_val = loadBatch("data_batch_2")
    X_test, Y_test, y_test = loadBatch("test_batch")

    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    for i in range(n_epochs):
        for j in range(0, N, n_batch):
            grad_W, grad_b = computeGradients(X[:,j:j+n_batch],Y[:,j:j+n_batch],W,l,b)
            W -= eta*grad_W
            b -= eta*grad_b
        
        P = evaluateClassifier(X,W,b)
        train_acc.append(computeAccuracy(P,y))
        train_loss.append(computeCost(P,Y,W,l))

        P_val = evaluateClassifier(X_val,W,b)
        val_acc.append(computeAccuracy(P_val,y_val))
        val_loss.append(computeCost(P_val,Y_val,W,l))

    P_test = evaluateClassifier(X_test,W,b)
    test_acc = computeAccuracy(P_test,y_test)
    test_loss = computeCost(P_test,Y_test,W,l)

    return W, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss

def run(l, n_batch, eta, n_epochs):
    W, b = initialize()
    W, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = miniBatch(W,b,l,n_epochs,n_batch,eta)

    print("Final test accuracy: " + str(test_acc*100) + " %")
    print("Final test loss: " + str(test_loss))
    
    plt.rcParams['figure.dpi'] = 100
    
    plt.figure(1)
    plt.plot(train_loss, "r-", label="Training Data")
    plt.plot(val_loss, "b-", label="Validation Data")
    plt.title("Loss Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid("true")
    plt.show()

    plt.figure(2)
    plt.plot(train_acc, "r-", label="Training Data")
    plt.plot(val_acc, "b-", label="Validation Data")
    plt.title("Accuracy Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid("true")
    plt.show()

    plt.figure(3)
    for i, j in enumerate(W):
        plt.subplot(2, 5, i+1)
        plt.imshow(np.rot90(np.reshape((j - j.min()) / (j.max() - j.min()), (32, 32, 3), order='F'), k=3))
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    X,Y,y = loadBatch("data_batch_1")
    W1,W2,b1,b2 = initialize()
    compareGradients(X[:20,0:2], Y[:,0:2], W1[:,0:20], W2, b1, b2, 0)
    # compareGradients(X[:500,0:100], Y[:,0:100], W[:,:500],l,b)
    # run(0,100,0.01,40)

