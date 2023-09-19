#!/usr/bin/env python
# coding: utf-8

# # Linear Regression
# $$ y = w*x + b $$

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


# # Simulating random data
# # Loss Function - Mean Squared Error
# $$ MSE = 1/N sum(y - yhat)**2   $$
# # R square

# In[5]:


def line(b, w, x):
    return b+w*x

"""Random data are generated in the get_data function"""

def get_data():
    
    np.random.seed(6)
    b = 4
    w = 2
    x = np.linspace (0,10,100)
    y = line(b, w, x)
    return x,y, b, w
    
    
    
    
if __name__ == "__main__":
    
    n_sim = 10    
    df = pd.DataFrame(columns = ["sim_id","b", "w", "delta_b", "delta_w", "MSE", "R_square"])
    
    
    db = []
    dw = []
    z = []
    L=[]
    Sigma = []
    r2 = []
    Y=[]
    Yhat=[]
    for i in range (0, n_sim):
        
        sg = float(i)
        np.random.seed(seed = 3* i+7)
        x,y,b,w = get_data()
        print(x.shape, y.shape)
        noise = np.random.normal(0.0, sg,[len(x)])
        y = y+noise
         
        x_p = np.c_[np.ones((100,1)), x]
        x_p.shape
        theta = np.linalg.inv(x_p.T.dot(x_p)).dot(x_p.T).dot(y)
        
        yhat = theta[0] + theta[1] * x
        
        loss =  np.sum ((y-yhat)** 2)
      
        delta_b = b - theta[0]
        delta_w = w - theta[1]
        
        db.append (delta_b)
        dw.append (delta_w)
        z.append(i)
        L.append(loss)
        Sigma.append(sg)
        y_bar = np.sum(y)/len(y)
        
        Y.append(y)
        Yhat.append(yhat)
        
        R_square = 1 - (np.sum((y-yhat)** 2))/np.sum(((y-y_bar)** 2))
        print(R_square)
        
        r2.append(R_square)
        
        
        plt.plot(x,y,"ro", label = "Data")
        plt.plot(x, yhat, 'b-', label = "Model")
        plt.plot(x,line(b,w,x),"k-", label = "Original line")
        plt.title("Noise = " +str(i))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("LR_plot",dpi = 400)
        plt.legend()
        plt.show()
        
        print(i, theta[0], theta[1], delta_b, delta_w, loss)
        
        
        df.loc[i] = [i, theta[0], theta[1], delta_b, delta_w, loss, R_square]
       
    print(df)
    
    fig, axs = plt.subplots (3,1,figsize=(8,12))
    print(dw, db)
    axs[0].plot(Sigma,L,label = "Loss- MSE")
    axs[0].set_xlabel("sigma")
    axs[0].set_ylabel("MSE loss")
    
   
    
    axs[1].plot(Sigma,dw)
    axs[1].set_xlabel("sigma")
    axs[1].set_ylabel("dw")
   
    
    axs[2].plot(Sigma,r2,label = "R_square")
    axs[2].set_xlabel("sigma")
    axs[2].set_ylabel("R_square")
    plt.legend()
    plt.show()
#     df.to_csv(r"C:\Users\Saranya.Sakkarapani\LR_Results_loss.csv")
    
    
    
    
    
        


# In[ ]:




