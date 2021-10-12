import sys, getopt
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
import scipy.optimize
from matplotlib import cm
from functools import partial
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
from scipy.special import softmax
from scipy.special import log_softmax
from scipy import optimize


def get_means(dimension,K,model='simplex'):
    mean = np.zeros((dimension,K))
    if(model=='simplex'):
        mean[:K,:]=np.sqrt(K/(K-1)) *(np.identity(K)-np.ones((K,K))/K)
    else:
        mean=np.random.normal(0,1,(dimension,K))
        for i in range(K):
            mean[:,i]=mean[:,i]/np.linalg.norm(mean[:,i])
    return mean

def get_samples(sample_complexity, mean, delta,prob):
    dimension = mean.shape[0]
    K = prob.shape[0]
    samples = int(sample_complexity * dimension)
    labels = np.random.choice(np.arange(0,K),samples,p=prob)
    Y=[]
    if(K>2):
        Y = np.zeros((K, samples))
    else:
        Y = -1+2*labels
    X = np.zeros((samples,dimension))
    for i in range(samples):
        if(K>2):
            Y[labels[i],i]=1
        X[i,:]=mean[:,labels[i]]
    #Means are order of 1
    X = X + np.sqrt(delta) * np.random.normal(0,1, (samples,dimension))
    X.reshape(samples,dimension)
    if(K>2):
        Y.reshape(K,samples)
    return X/np.sqrt(dimension), Y, labels


# Implements pseudo-inverse solution for ridge regression
def ridge_estimator(X, Y, lamb=0.1):
    '''
    X has shape n x d
    Y has shape K x n
    returns: W with shape K x d
    '''
    n, d = X.shape
    if n >= d:
        W = np.linalg.inv(X.T @ X + lamb*np.identity(d)) @ X.T @ Y.T
    elif n < d:
        W = X.T @ np.linalg.inv(X @ X.T + lamb*np.identity(n)) @ Y.T
    return W.T

# Simulate the problem for a given sample complexity and average over given number of seeds.

def simulate(K,sample_complexity, lamb, prob, delta, loss,model,seeds=1, dimension=1000):
    eg, et, eth = [], [], []

    # Average over seeds
    for i in range(seeds):
        mean=get_means(dimension,K,model)
        mm2= mean @ mean.T
        X_train, Y_train, lab_train = get_samples(sample_complexity, mean, delta,prob)
        X_test,  Y_test,  lab_test  = get_samples(sample_complexity, mean, delta,prob)

        W=np.zeros((K,dimension))
        test_error=0
        #train_error=0
        train_error_ham=0
        Y_test_hat=0

        if(loss=='qua'):
            sam=X_train.shape[0]
            X_train = np.hstack((X_train,np.ones((sam,1))))
            sam=X_test.shape[0]
            X_test = np.hstack((X_test,np.ones((sam,1))))
            U = ridge_estimator(X_train, Y_train, lamb=lamb)
            W=[]
            Y_train_hat = U @ X_train.T
            Y_test_hat = U @ X_test.T
            #train_error = 0.5* np.mean((Y_train_hat - Y_train)**2)
            if(K>2):
                W=U[:,:dimension]
                test_error = zero_one_loss(np.argmax(Y_test, axis=0), np.argmax(Y_test_hat, axis=0))
                train_error_ham = zero_one_loss(np.argmax(Y_train, axis=0), np.argmax(Y_train_hat, axis=0))
            else:
                W=U[:dimension]
                test_error = zero_one_loss(Y_test, np.sign(Y_test_hat))
                train_error_ham = zero_one_loss(Y_train, np.sign(Y_train_hat))
        elif(loss=='log'):
            if(lamb>0):
                clf = LogisticRegression(penalty='l2',C=1./lamb, fit_intercept=True,multi_class='multinomial',solver='lbfgs',random_state=0,max_iter=10000, tol=0.00001).fit(X_train,lab_train)
            else:
                clf = LogisticRegression(penalty='none', fit_intercept=True,multi_class='multinomial', random_state=0,max_iter=5000).fit(X_train,lab_train)
            W=clf.coef_
            b=clf.intercept_
            u=W @ X_train.T+b[:,np.newaxis]
            if(K>2):
                Y_train_hat = -log_softmax(u, axis=0)
                #train_error = np.einsum('is,is->s',Y_train,Y_train_hat).mean()
            #else:
                #train_error = np.log(1+np.exp(-lab_train * u)).mean()
            lab_train_hat = clf.predict(X_train)
            lab_test_hat = clf.predict(X_test)
            train_error_ham = 1-(lab_train==lab_train_hat).mean()
            test_error = 1-(lab_test==lab_test_hat).mean()
        else:
            raise NotImplementedError

        Q = delta*W @ W.T/dimension
        m =  W @ mean/np.sqrt(dimension)

        eg.append(test_error)
        #et.append(train_error)
        eth.append(train_error_ham)

    return (np.mean(eth), np.mean(eg), np.std(eth),np.std(eg))
