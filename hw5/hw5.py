
# coding: utf-8

# In[47]:

import numpy as np
import matplotlib.pyplot as plt


# In[48]:

class svmclass:
    def __init__(self,data,label):
        self.trainX = data
        self.trainY = label
        self.alpha  = np.ones([label.shape[0],1])
        self.bias = 0
        self.shift = np.zeros([1,data.shape[1]])
        self.scalefactor = np.zeros([1,data.shape[1]])
        self.sv = data
        self.sv_idx = np.array(range(data.shape[0]))
        self.kernelfunc = rbf_kernel


# In[49]:

def linear_kernel(x,y):
    return np.dot(x,y.T)

def poly_kernel(x,y):
    gamma = 0.5
    r = 1
    d = 3
    return (gamma*np.dot(x,y.T)+r)**d

def rbf_kernel(x,y):
    sigma = 0.5
    return np.exp(-1/(2*sigma**2)*np.array([(x**2).sum(1)]).T+np.array([(y.T**2).sum(0)]-2*np.dot(x,y.T)))
    
def exp_kernel(x,y):
    sigma = 0.5
    kval = np.zeros([x.shape[0],y.shape[0]])
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            kval[i,j] = abs(x[i,:]-y[j,:]).sum(0)
    return np.exp(-1/(2*sigma**2)*kval)


# In[50]:

def trainsvm(data,label):
    # svm for binary case
    label[label<0] = -1
    label[label>0] = 1
    
    kernel = exp_kernel
    N,D = data.shape
    BoxC = np.ones([N,1])
    
    # LS
    
    shift = data.mean(0)
    scalefactor = 1.0 / data.std(0)
    for i in range(D):
        data[:,i] = (data[:,i]-shift[i])*scalefactor[i]
    
    #kernel
    Omega = kernel(data,data)
    Omega = 0.5*(Omega+Omega.T) + np.diag(BoxC[:,0])
    
    #Hessian
    Q = np.dot(label,label.T)*Omega
    
    #solve Ax=b
    A = np.matrix(np.zeros([N+1,N+1]))
    A[1:,0] = np.matrix(label)
    A[0,1:] = np.matrix(-label.T)
    A[1:,1:] = np.matrix(Q)
    
    b = np.matrix(np.ones([N+1,1]))
    b[0] = 0
    x = np.linalg.solve(A,b)
    bias = x[0]
    alpha = label*np.array(x[1:])
    sv = data
    sv_idx = np.array(range(N))

    svm = svmclass(data,label)
    svm.kernelfunc = kernel
    svm.scalefactor = scalefactor
    svm.shift = shift
    svm.alpha = alpha
    svm.bias = bias
    svm.sv = sv
    svm.sv_idx = sv_idx
    
    return svm


# In[51]:

def classify(svm,testX):
    for i in range(testX.shape[1]):
        testX[:,i] = svm.scalefactor[i]*(testX[:,i]-svm.shift[i])
    
    pred = np.dot((svm.kernelfunc(svm.sv,testX)).T,svm.alpha)+svm.bias
    return np.sign(pred)


# In[95]:

def main():
    num = 100
    rad1 = np.sqrt(np.random.rand(num,1))
    ang1 = 2*np.pi*np.random.rand(num,1)
    data1 = np.zeros([num,2])
    data1[:,0] = (rad1*np.cos(ang1))[:,0]
    data1[:,1] = (rad1*np.sin(ang1))[:,0]
    
    rad2 = np.sqrt(3*np.random.rand(num,1)+1)
    ang2 = 2*np.pi*np.random.rand(num,1)
    data2 = np.zeros([num,2])
    data2[:,0] = (rad2*np.cos(ang2))[:,0]
    data2[:,1] = (rad2*np.sin(ang2))[:,0]
    
    fig1 = plt.figure(1)
    plt.grid(True)
    h1 = plt.plot(data1[:,0],data1[:,1],'bo',label="1(train)")
    h2 = plt.plot(data2[:,0],data2[:,1],'ro',label="2(train)")
    
    cir = plt.Circle((0,0),1,facecolor='none',edgecolor='g',linewidth=2,alpha=0.5)
    plt.gca().add_patch(cir)
    cir = plt.Circle((0,0),2,facecolor='none',edgecolor='g',linewidth=2,alpha=0.5)
    plt.gca().add_patch(cir)
    plt.axis('equal')
    
    traindata = np.concatenate((data1,data2))
    label = np.ones([2*num,1])
    label[num:2*num] = -1
    
    testnum = 100
    rad0 = np.sqrt(4*np.random.rand(testnum,1))
    ang0 = 2*np.pi*np.random.rand(testnum,1)
    testdata = np.zeros([testnum,2])
    testdata[:,0] = (rad0*np.cos(ang0))[:,0]
    testdata[:,1] = (rad0*np.sin(ang0))[:,0]
    plt.plot(testdata[:,0],testdata[:,1],'bo',mfc='none')
    
    svm = trainsvm(traindata,label)
    labelY = classify(svm,testdata)
    idx1 = np.squeeze(np.array(labelY))==1
    idx2 = np.squeeze(np.array(labelY))==-1

    plt.plot(testdata[idx1,0],testdata[idx1,1],'bx',label="1(classified)")
    plt.plot(testdata[idx2,0],testdata[idx2,1],'rx',label="2(classified)")
    plt.legend()
    
    truelabel = np.ones([testnum,1])
    for i in range(testnum):
        dist = np.sqrt((testdata[i,:]**2).sum())
        if dist > 1:
            truelabel[i] = -1
            
    accuracy = (testnum-0.5*(abs(truelabel-labelY)).sum())/testnum
    plt.title('exp - Accuracy = %.2f'%accuracy)
    plt.show()
    fig1.savefig("hw5_exp_rbf_classified.pdf")


# In[105]:

if __name__=='__main__':
    main()

