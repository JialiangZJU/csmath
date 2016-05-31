
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt


# In[2]:

def Load_data(filename):
    f = open(filename)
    
    data = []
    num = 0
    
    for line in f:
        Line = map(int,line.split(','))
        data.append(Line)
        num += 1
        
    data = np.array(data)
    t = data[:,0:-1] / 255.0
    t_label = data[:,-1]
    
    # t : Num * dimension
    return t, t_label


# In[3]:

def PCA(data,p):
    # Input :
    #    data    -Data Matrix. Each row vector of fea is a data point
    #    p       -Number of dimension to project
    
    # Output :
    #    data_mean
    #    eigvector
    #    eigvalue
    data_mean = data.mean(0)
    
    eigvalue,eigvector = np.linalg.eig(np.dot((data-data_mean).T,data-data_mean))
    
    return data_mean, eigvector[:,0:p], eigvalue[0:p]
    


# In[4]:

def Project(data_0_mean,eigvector,eigvalue):
    return np.dot(data_0_mean,eigvector)*eigvalue


# In[5]:

def plot_eig(data_mean,eigvector,eigvalue,digit):
    fig1 = plt.figure(1,figsize=(12, 5))
    plt.subplot(131)
    plt.xlabel('mean')
    plt.imshow(data_mean.reshape(8,8),cmap=plt.cm.gray)
    
    plt.subplot(132)
    plt.xlabel('1st Component')
    plt.title('$\lambda = %.3f$' % eigvalue[0])
    plt.imshow(eigvector[:,0].reshape(8,8),cmap=plt.cm.gray)
    
    plt.subplot(133)
    plt.xlabel('2nd Component')
    plt.title('$\lambda = %.3f$' % eigvalue[1])
    plt.imshow(eigvector[:,1].reshape(8,8),cmap=plt.cm.gray)
    
    plt.subplots_adjust(wspace=0.2, hspace=0.2, top=1, bottom=0, left=0.1, right=0.9)
    plt.show()
    
    fig1.savefig("mean_compoent_%d.pdf"%digit);


# In[6]:

def feat_point(x,y,p):
    N = len(x)
    interval_x = (max(x)-min(x))/float(p)
    interval_y = (max(y)-min(y))/float(p)
    ref_x = map(lambda t:min(x)+interval_x*(t+0.5),range(p))
    ref_y = map(lambda t:min(y)+interval_y*(t+0.5),range(p))
    p_indx = []   
    for i in range(p):
        for j in range(p):
            f = lambda a:(ref_x[i]-x[a])**2+(ref_y[j]-y[a])**2
            dist = map(f,range(N))      
            ind = dist.index(min(dist))
            p_indx.append(ind)
    return p_indx, ref_x, ref_y


# In[7]:

def plot_images_matrix(train,indx,digit):
    p = int(np.sqrt(len(indx)))   
    fig3 = plt.figure(3,figsize=(10, 9))
    for i in range(p):
        for j in range(p):
            plt.subplot(p,p,p*(p-i-1)+j+1)
            img = train[indx[i*p+j],:].reshape(8,8)
            plt.imshow(img,cmap=plt.cm.gray)
            plt.axis('off')
            #plt.grid(True)
    plt.subplots_adjust(wspace=0.01, hspace=0.02, top=0.95, bottom=0.05, left=0.05, right=0.95)
    plt.show()
    fig3.savefig('images_matrix_%d.pdf'%digit)


# In[8]:

def main():
    digit = 3
    dimension = 2
    train,train_label = Load_data('ORHDDS/optdigits.tra')
    train = train[train_label == digit,:]
    train_label = train_label[train_label == digit]
    
    dm,eigvec,eigval = PCA(train,2)
    plot_eig(dm,eigvec,eigval,digit)
    
    Proj = Project(train-dm,eigvec,eigval)
    Proj = Proj * 100
    
    p = 5
    indx,ref_x,ref_y = feat_point(Proj[:,0],Proj[:,1],p)
    
    fig2 = plt.figure(2)
    plt.xlabel('1st Component')
    plt.ylabel('2nd Component')
    plt.plot(Proj[:,0],Proj[:,1],'o',color='#00FF00')
    
    plt.axhline(y=0,color='black')
    plt.axvline(x=0,color='black')
    p = int(np.sqrt(len(indx)))
    for i in range(p):
        plt.axvline(ref_x[i],linestyle='--',color='.5')
        plt.axhline(ref_y[i],linestyle='--',color='.5')

    plt.plot(Proj[:,0][indx],Proj[:,1][indx],'ro')
    plt.show()
    fig2.savefig('feature_space_%d.pdf'%digit)

    plot_images_matrix(train,indx,digit)


# In[9]:

if __name__=='__main__':
    main()

