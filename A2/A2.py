import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

N = 10000
d = 3072
k = 10
m = 50
eps = np.finfo(float).eps
eta_min = 1e-5
eta_max = 1e-1
n_batch = 100

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

def cycleETA(t, l_cycles, n_s, eta):
    eta_lower = 2*l_cycles*n_s
    eta_mid = (2*l_cycles + 1) * n_s
    eta_upper = 2*(l_cycles + 1) * n_s
    eta_range = eta_max - eta_min

    if (eta_lower <= t) and (t <= eta_mid):
        eta = eta_min + eta_range*(t-eta_lower)/n_s
    elif (eta_mid <= t) and (t <= eta_upper):
        eta = eta_max - eta_range*(t-eta_mid)/n_s
    
    return eta

def getData(data_set):
    if data_set == "big":
        X,Y,y = loadBatch("data_batch_1")
        for i in range(2,6):
            X_temp, Y_temp, y_temp = loadBatch("data_batch_" + str(i))
            X = np.append(X,X_temp,axis=1)
            Y = np.append(Y,Y_temp,axis=1)
            y = np.append(y,y_temp,axis=0)
        X,X_val = np.split(X,[45000],axis=1)
        Y,Y_val = np.split(Y,[45000],axis=1)
        y,y_val = np.split(y,[45000],axis=0)
    else:
        X, Y, y = loadBatch("data_batch_1")
        X_val, Y_val, y_val = loadBatch("data_batch_2")
    
    X_test, Y_test, y_test = loadBatch("test_batch")

    return X, Y, y, X_val, Y_val, y_val, X_test, Y_test, y_test
    
def miniBatch(W1, W2, b1, b2, l, eta, n_s, n_cycles, data_set):
    X, Y, y, X_val, Y_val, y_val, X_test, Y_test, y_test = getData(data_set)
    print(X.shape)
    print(Y.shape)
    print(y.shape)
    print(X_val.shape)
    print(Y_val.shape)
    print(y_val.shape)

    train_acc = []
    train_loss = []
    train_cost = []
    val_acc = []
    val_loss = []
    val_cost = []
    iterations = []

    t = 0
    l_cycles = -1

    n_epochs = math.ceil(2*n_cycles*n_s/(X.shape[1]/n_batch))

    for i in range(n_epochs):
        for j in range(0, X.shape[1], n_batch):
            grad_W1, grad_W2, grad_b1, grad_b2 = computeGradients(X[:,j:j+n_batch],Y[:,j:j+n_batch],W1,W2,b1,b2,l)
            W1 -= eta*grad_W1
            W2 -= eta*grad_W2
            b1 -= eta*grad_b1
            b2 -= eta*grad_b2

            if t % (2 * n_s) == 0:
                l_cycles += 1
            
            eta = cycleETA(t,l_cycles,n_s, eta)
            t += 1

            if(t % 100 == 0):
                iterations.append(t)
                P,_ = evaluateClassifier(X,W1,W2,b1,b2)
                P_val, _ = evaluateClassifier(X_val,W1,W2,b1,b2)
        
                train_acc.append(computeAccuracy(P,y))
                val_acc.append(computeAccuracy(P_val,y_val))

                temp_loss, temp_cost = computeCost(P,Y,W1,W2,l)
                train_loss.append(temp_loss)
                train_cost.append(temp_cost)

                temp_loss, temp_cost = computeCost(P_val,Y_val,W1,W2,l)
                val_loss.append(temp_loss)
                val_cost.append(temp_cost)

    P_test,_ = evaluateClassifier(X_test,W1,W2,b1,b2)
    test_acc = computeAccuracy(P_test,y_test)
    test_loss = computeCost(P_test,Y_test,W1,W2,l)

    return train_acc, train_loss, train_cost, val_acc, val_loss, val_cost, test_acc, test_loss, iterations

def run(l, eta, n_s, n_cycles, data_set):
    W1,W2,b1,b2 = initialize()
    train_acc, train_loss, train_cost, val_acc, val_loss, val_cost, test_acc, test_loss, iterations = miniBatch(W1,W2,b1,b2,l,eta,n_s,n_cycles,data_set)

    print("Final test accuracy: " + str(test_acc*100) + " %")
    print("Final test loss: " + str(test_loss))
    
    plt.rcParams['figure.dpi'] = 100
    
    plt.figure(1)
    plt.plot(iterations, train_cost, "r-", label="Training Data")
    plt.plot(iterations, val_cost, "b-", label="Validation Data")
    plt.title("Cost")
    plt.xlabel("Update Step")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid("true")
    plt.show()

    plt.figure(2)
    plt.plot(iterations, train_loss, "r-", label="Training Data")
    plt.plot(iterations, val_loss, "b-", label="Validation Data")
    plt.title("Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid("true")
    plt.show()

    plt.figure(3)
    plt.plot(iterations, train_acc, "r-", label="Training Data")
    plt.plot(iterations, val_acc, "b-", label="Validation Data")
    plt.title("Accuracy")
    plt.xlabel("Update Step")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid("true")
    plt.show()

if __name__ == "__main__":
    # X,Y,y = loadBatch("data_batch_1")
    # W1,W2,b1,b2 = initialize()
    # compareGradients(X[:20,0:2], Y[:,0:2], W1[:,0:20], W2, b1, b2, 0)
    # # compareGradients(X[:500,0:100], Y[:,0:100], W[:,:500],l,b)
    run(0.01,0.01,800,3,"big")

