
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# In[2]:

def obj_func(x):
    func = np.array([10*(x[1]-x[0]**2),1-x[0]])
    y = np.dot(func,func)
    J = np.array([[-20*x[0],10],[-1,0]])
    grad = np.dot(J.T,func)
    return y,grad,J


# In[3]:

def LM(_func,x0,eps = 10**(-7),maxIter = 1000):
    D = len(x0)
    Lambda = 0.01
    y,grad,J = _func(x0)
    Iter = 0
    x = x0
    
    while Iter<maxIter and np.linalg.norm(grad)>eps:
        Iter += 1
        y,grad,J = _func(x)
        H = np.dot(J.T,J)+Lambda*np.eye(D)
        eigval,eigvec = np.linalg.eig(H)
        # positive definition (not semi)
        while ~(eigval>np.zeros(D)).all():
            Lambda *= 4
            H = np.dot(J.T,J)+Lambda*np.eye(D)
            eigval,eigvec = np.linalg.eig(H)
            
        Iter2 = 0
        while Iter2<maxIter:
            Iter2 += 1
            
            Dx,residuals,rank,s =np.linalg.lstsq(H,-grad)
            xnew = x + Dx
            ynew,gradnew,Jnew = _func(xnew)
            #print ynew
            delta_y = np.linalg.norm(ynew-y)
            delta_q = np.linalg.norm(np.dot(grad.T,xnew-x))
            rho = delta_y/delta_q
            
            if rho > 0:
                x = xnew
                if rho < 0.25:
                    Lambda *= 4
                elif rho > 0.75:
                    Lambda /= 2
                #print y,x,np.linalg.norm(grad),Lambda
                break
            else:
                Lambda *= 4
                #print y,x,np.linalg.norm(grad),Lambda
                continue
    return x


# In[4]:

def main():
    x = np.array([-100,100])
    result = LM(obj_func,x)
    print 'result = ',result


# In[5]:

if __name__=='__main__':
    main()

