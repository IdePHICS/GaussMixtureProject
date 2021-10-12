import sys, getopt
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
import scipy.optimize
from matplotlib import cm
from functools import partial
from scipy.special import softmax
from scipy.special import log_softmax
from scipy import optimize


### Generic tools
def damping(q_new, q_old, coef_damping=0.7):
    return (1 - coef_damping) * q_new + coef_damping * q_old

itera=50000 #Montecarlo steps
IDEK=[]
IDE=[]
K=0

def get_means(K,model='simplex'):
    mean = []
    if(model=='simplex'):
        mean=np.sqrt(K/(K-1)) *(np.identity(K)-np.ones((K,K))/K)
        return mean.T @ mean
    else:
        return np.identity(K)

### Logistic loss functions

def get_eta(omega,V,lamb,it=itera):
    K=V.shape[0]
    ide = np.tensordot(np.identity(K),np.ones(it),axes=0)
    f = ide
    la=lamb
    def Dl(f):
        f.reshape(K,K,it)
        Vfo=np.einsum('ij,jks->iks',V,f)+omega
        expf = softmax(Vfo/la,axis=0)
        return f - ide + expf
    if(lamb>0.001):
        f = scipy.optimize.root(Dl,ide,method='df-sane').x
    else:
        valori=[0.001*np.exp(-i) for i in range(5)]
        for ll in valori:
            la=max(ll,lamb)
            f = scipy.optimize.root(Dl,f,method='df-sane').x
            if(lamb>ll):
                break
    return f

def lxx(v):
    K=v.shape[0]
    soft=softmax(v,axis=0)
    return - np.einsum('iku,jku->ijku',soft,soft) + np.einsum('iku,ij->ijku',soft,np.identity(K))

def l(u,v):
    soft = log_softmax(v,axis=0)
    return - np.einsum('ik...,ik...->k...',u,soft)


def update_hat_log(V, Q, m, b, alpha, lamb, delta, prob):
    K=prob.shape[0]
    sqQ= np.real(sqrtm(Q))
    omega = np.random.normal(0,1,(K,K,itera))
    omega = np.einsum('ij,jsk -> isk',sqQ, omega) + (m+b)[:,:,np.newaxis]
    f=get_eta(omega,V,lamb)
    Vfo=np.einsum('ij,jsk -> isk',V,f)+omega
    lXX=lxx(Vfo/lamb)/lamb
    Inversa=(lXX.transpose(3,2,0,1) @ V).transpose(2,3,1,0)+IDEK
    Inversa=np.linalg.inv(Inversa.T).T
    mh0=np.einsum('k,iks->iks',prob,f)
    MH = alpha * mh0.sum(axis=2).T*1./itera
    QH = alpha * delta * np.einsum('iks,jks ->ij',f,mh0)*1./itera
    VH = alpha * delta * np.einsum('k,iuks,ujks->ij',prob,Inversa, lXX)*1./itera
    b = np.einsum("k,iks->is",prob,Vfo-m[:,:,np.newaxis]).mean(axis=1)
    return QH, MH, VH, b

### Quadratic loss functions
def update_hat_qua(V, Q, m, b, alpha, lamb, delta, prob):
    K=prob.shape[0]
    e = np.identity(K)
    diff = m-e+b.reshape(K,1) @ np.ones((1,K))
    VH = np.linalg.inv(e+V)
    QH = alpha * delta * VH @ (Q + diff @ np.diag(prob) @ diff.T) @ VH
    MH = - alpha * VH @ (prob * diff)
    VH = alpha * delta * VH
    b =(prob*(e-m)).sum(axis=1)
    return QH, MH, VH, b

### General

def update_sp(V, Q, m, b, alpha, lamb, delta, mumu, prob, loss):
    '''
    One step update of saddle-point equations
    '''
    K=prob.shape[0]
    Qh, Vh, Mh = [], [], []
    
    if(loss=='qua'):
        Qh, Mh, Vh, b = update_hat_qua(V, Q, m, b, alpha, lamb, delta, prob)
        inv_hat = np.linalg.inv(lamb*np.identity(K) + Vh)
    elif(loss=='log'):
        Qh, Mh, Vh, b = update_hat_log(V, Q, m, b, alpha, lamb, delta, prob)
        inv_hat = np.linalg.inv(np.identity(K) + Vh)
    else:
        raise NotImplementedError
    
    Vnew = delta * inv_hat
    Qnew = delta * inv_hat @ (Qh + Mh @ mumu @ Mh.T) @ inv_hat
    Mnew = inv_hat @ Mh @ mumu.T

    return Qnew, Mnew, Vnew, b

def iterate(alpha, lamb, delta, mumu, prob, loss, max_iter=1000,eps=0.00001,qi=np.eye(K),Vi=np.eye(K),mi=np.ones(K),bi=np.zeros(K)):
    # Initialise
    K=prob.shape[0]
    q = np.zeros((max_iter, K, K))
    m = np.zeros((max_iter, K, K))
    V = np.zeros((max_iter, K, K))
    b = np.zeros((max_iter, K))

    global IDE
    IDE=np.tensordot(np.identity(K),np.ones(itera),axes=0)
    global IDEK
    IDEK=np.tensordot(np.identity(K),np.ones((K,itera)),axes=0)

    q[0], m[0], V[0], b[0] = qi, mi, Vi, bi

    for t in range(max_iter - 1):
        qtmp, mtmp, Vtmp, btmp = update_sp(V[t], q[t], m[t], b[t], alpha, lamb, delta, mumu, prob, loss)
        q[t + 1], m[t+1], V[t+1], b[t+1] = damping(qtmp, q[t]), damping(mtmp, m[t]), damping(Vtmp, V[t]), damping(btmp, b[t])
        diff = np.linalg.norm(q[t+1]-q[t])+np.linalg.norm(m[t+1]-m[t])+np.linalg.norm(V[t+1]-V[t])+np.linalg.norm(b[t+1]-b[t])
        if diff < eps:
            break
    train_error=0
    train_error_ham=0
    test_error=0
    Qf=q[t+1]
    Mf=m[t+1]
    Vf=V[t+1]
    bf=b[t+1]

    sqQ=np.real(sqrtm(Qf))
    omega = np.random.normal(0,1,(K,K,1000000))
    omega = np.einsum('ij,jsk -> isk',sqQ, omega) + (Mf+bf)[:,:,np.newaxis]
    aux=np.tensordot(np.arange(K),np.ones(1000000),axes=0)
    if loss == 'log':
        f=get_eta(omega,Vf,lamb,it=1000000)
        eta=np.einsum('ij,jks->iks',Vf,f)+omega
        #train_error=prob @ (l(np.eye(K)[:,:,np.newaxis],eta/lamb).mean(axis=1))
        lamLoss=(np.argmax(eta,axis=0)==aux).mean(axis=1)
        train_error_ham=1-prob @ lamLoss
    elif loss=='qua':
        eta = omega+ Vf[:,:,np.newaxis]
        eta = np.einsum('ij,jks -> iks',np.linalg.inv(np.eye(K)+Vf),eta)
        lamLoss=(np.argmax(eta,axis=0)==aux).mean(axis=1)
        train_error_ham=1-prob @ lamLoss
        #train_error=.5*prob @ ((np.linalg.norm(np.eye(K)[:,:,np.newaxis]-eta,axis=0)**2).mean(axis=1))
    else:
        raise NotImplementedError

    kk = np.arange(K)
    Lam=np.zeros((1000000,K,K))
    for k in range(K):
        Lam[:,:,k] = np.random.multivariate_normal(Mf[:,k]+bf,Qf,1000000)
    yhat=np.argmax(Lam,axis=1)
    diff = np.equal(yhat,kk).mean(axis=0)
    test_error = prob @ diff
    test_error = 1 - test_error
    return Qf, Mf, Vf, bf, t+1, test_error, train_error_ham
