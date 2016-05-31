
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import sys


# In[2]:

def GenerateData_sin(num):
    # gererate data (x,y) x varies from 0 to 1
    # y = sin(x)+epsilon
    # gaussing noise
    x = np.linspace(0,1,num)
    
    mu = 0
    sigma = 0.15
    gaussing_noise = np.random.normal(mu,sigma,num)
    
    y = np.sin(x*2*np.pi) + gaussing_noise
    
    return x, y


# In[3]:

def PolyFit(x,y,D,Lambda):
    # Least squares
    # Lambda for ridge regression
    # D for degree
    N = x.size
    A = np.ones([N,D+1])
    
    for i in range(0,D):
        A[:,D-i-1] = A[:,D-i]*x
    
    b = np.dot(np.linalg.inv(np.dot(A.T,A)+Lambda*np.eye(D+1)),(np.dot(A.T,y)))
    
    return b


# In[4]:

def main(Num,D):
    #Num, number of data point
    #D, degree
    x, y = GenerateData_sin(Num)
 
    fig = plt.figure(1)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([-0.1,1.1,-1.5,1.5])
    
    plt.plot(x,y,'bo',mew=2,mec='b',mfc='none',ms=8)
    
    # plot sin(x)
    x_sin = np.linspace(0,1,1000)
    plt.plot(x_sin,np.sin(2*x_sin*np.pi),'#00FF00',lw=2,label='$y = \sin(x)$')
    
    # plot PolyFit curve
    Lambda = np.exp(-18)
    #Lambda = 0
    b = PolyFit(x,y,D,Lambda)
    
    PolyFit_curve = np.poly1d(b)
    x_fit = np.linspace(0,1,1000)
    #label ='$N = ' + str(Num) + ', D = ' + str(D) + ', \Lambda = '+ str(Lambda) + '$'
    label ='$N = ' + str(Num) + ', D = ' + str(D) + ', \Lambda = exp(-18)$'
    plt.plot(x_fit,PolyFit_curve(x_fit),'-r',label=label,lw=2)

    plt.legend()
    plt.show()
    figname = 'PolyFit_D'+str(D)+'_N'+str(Num)+'Reg.pdf'
    fig.savefig(figname)


# In[5]:

if __name__=='__main__':
    print sys.argv[1]
    
    main(int(sys.argv[1]),int(sys.argv[2]))

