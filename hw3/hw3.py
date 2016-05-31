
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import random


# In[2]:

def kmeans(X,k,threshold=1e-15,maxiter=300):
    N = len(X)
    labels = np.zeros(N, dtype=int)
    centers = np.array(random.sample(X,k))
    Iter = 0
    
    def calc_J():
        sum = 0
        for i in xrange(N):
            sum += np.linalg.norm(X[i]-centers[labels[i]])
        return sum
    
    def distmat(X,Y):
        n = len(X)
        m = len(Y)
        result = np.zeros([n,m])
        xx = (X*X).sum(1)
        yy = (Y*Y).sum(1)
        xy = np.dot(X,Y.T)
        result += np.array(np.matrix(xx).T)
        result += yy
        result -= 2*xy
        return result
    
    Jprev = calc_J()
    
    while True:
        dist = distmat(X,centers)
        labels = dist.argmin(1)
        
        for j in range(k):
            idx_j = (labels==j).nonzero()
            centers[j] = X[idx_j].mean(0)
        
        J = calc_J()
        Iter += 1
        
        if Jprev-J < threshold:
            break
        Jprev = J
        if Iter >= maxiter:
            break
        
    return centers,labels


# In[3]:

def GMM(X,k):
    centers,labels = kmeans(X,k)
    N,D = X.shape
    
    def init_params():
        pMu = centers
        pPi = np.zeros(k)
        pSigma = np.zeros([k,D,D])
        
        for i in range(k):
            Xk = X[labels==i,:]
            pPi[i] = 1.0*len(Xk)/N;
            pSigma[i,:,:] = np.dot((Xk-centers[i]).T,Xk-centers[i])/len(Xk)
        return pMu,pPi,pSigma
    
    pMu,pPi,pSigma = init_params()
    
    def calc_prob():
        Px = np.zeros([N,k])
        for i in range(k):
            Xshift = X - pMu[i,:]
            inv_pSigma = np.linalg.inv(pSigma[i,:,:])
            tmp = (np.dot(Xshift,inv_pSigma)*Xshift).sum(1)
            coef = np.power(2*np.pi,-float(D)/2) * np.power(np.linalg.det(inv_pSigma),1.0/2)
            Px[:,i] = coef * np.exp(-0.5*tmp)
        return Px
    
    threshold = 1e-15
    Lprev = -1e100
    Maxiter = 100
    Iter = 0
    while Iter<Maxiter:
        Iter += 1
        
        Px = calc_prob()
        pGamma = Px * pPi
        pGamma = 1.0*pGamma / np.array(np.matrix(pGamma.sum(1)).T)
        
        Nk = pGamma.sum(0)
        pMu = np.dot(np.array(np.matrix(1.0/Nk).T)*pGamma.T,X)
        pPi = 1.0*Nk/N
        
        for kk in range(k):
            Xshift = X - pMu[kk,:]
            pSigma[kk] = np.dot(Xshift.T*pGamma[:,kk],Xshift)/Nk[kk]
        
        L = np.log(np.dot(Px,pPi.T)).sum()
        if L-Lprev<threshold:
            break
        Lprev = L
    
    return Px,pMu,pSigma,pPi


# In[4]:

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """ 
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


# In[5]:

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """ 
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


# In[6]:

def main():
    N = 200;
    weight = [0.25,0.25,0.25,0.25];

    mu = [0,0];
    cov = [[0.1,0],[0,1.5]];
    data1 = np.random.multivariate_normal(mu,cov,int(N*weight[0]))

    mu2 = [4,0];
    cov2 = [[0.1,0],[0,1.5]];
    data2 = np.random.multivariate_normal(mu2,cov2,int(N*weight[1]))

    mu3 = [2,2];
    cov3 = [[1.5,0],[0,0.1]];
    data3 = np.random.multivariate_normal(mu3,cov3,int(N*weight[2]))

    mu4 = [-2,-2];
    cov4 = [[1.5,0],[0,0.1]];
    data4 = np.random.multivariate_normal(mu4,cov4,int(N*weight[2]))
    
    fig1 = plt.figure(1)
    plt.grid(True)
    plt.plot(data1[:,0],data1[:,1],'go');
    plt.plot(data2[:,0],data2[:,1],'ro');
    plt.plot(data3[:,0],data3[:,1],'bo');
    plt.plot(data4[:,0],data4[:,1],'yo');
    
    plt.axis('equal');
    plt.show();
    fig1.savefig("Gauss_Distribution.pdf");

    data = np.array(data1.tolist() + data2.tolist() + data3.tolist() + data4.tolist());
    (alpha,Mu,Cov,w) = GMM(data,4)
    ind = alpha.argmax(1)

    fig2 = plt.figure(2)
    plt.grid(True)
    plt.plot(data[ind==0,0],data[ind==0,1],'go');
    plt.plot(data[ind==1,0],data[ind==1,1],'bo');
    plt.plot(data[ind==2,0],data[ind==2,1],'ro');
    plt.plot(data[ind==3,0],data[ind==3,1],'yo');
    plt.axis('equal');
    plot_cov_ellipse(Cov[0], Mu[0], nstd=3, alpha=0.5, color='green')
    plot_cov_ellipse(Cov[1], Mu[1], nstd=3, alpha=0.5, color='blue')
    plot_cov_ellipse(Cov[2], Mu[2], nstd=3, alpha=0.5, color='red')
    plot_cov_ellipse(Cov[3], Mu[3], nstd=3, alpha=0.5, color='yellow')
    plt.show();
    fig2.savefig("EM_MoGs.pdf");


# In[7]:

if __name__=='__main__':
    main()

