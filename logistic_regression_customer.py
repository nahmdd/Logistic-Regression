#!/usr/bin/env python
# coding: utf-8

# # imports

# In[2]:


import numpy as np 
import pandas as pd
from numpy import log,dot,exp,shape
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.metrics import accuracy_score,  confusion_matrix
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler


# # load data

# In[3]:


df = pd.read_csv("Downloads\customer_data.csv")
df 


# In[4]:


df=df.sample(frac=1)
df


# # preprocessing 

# In[5]:


df.isna().sum()


# In[6]:


X = df.drop(columns = ['purchased']) 
y = df['purchased']


# In[7]:


#checking outlires
fig = plt.figure(figsize=(10,10))
for index,col in enumerate(X.columns):
    plt.subplot(6,4,index+1)
    sns.boxplot(df.loc[:,col])


# In[8]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
(((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()/df.shape[0])*100


# In[9]:


num = df.duplicated().sum()
num


# In[10]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)


# In[11]:


x_train_array = X_train.to_numpy()
x_test_array = X_test.to_numpy()

y_train_array = y_train.to_numpy()
y_test_array = y_test.to_numpy()


# # logistic regression

# In[12]:



scaler = MinMaxScaler()
x_train_array = scaler.fit_transform(x_train_array)
x_test_array = scaler.fit_transform(x_test_array)


# In[13]:


class logistic_regression:
    def sigmoid(self,z):
        sig = 1/(1+exp(-z))
        return sig
    def initialize(self,X):
        weights = np.zeros((shape(X)[1]+1,1))  
        X = np.c_[np.ones((shape(X)[0],1)),X]
        return weights,X
    def fit(self,X,y,alpha=0.001,iter=200):
        weights,X = self.initialize(X)
        def cost(theta):
            z = dot(X,theta)
            cost0 = y.T.dot(log(self.sigmoid(z))) 
            cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
            cost = -((cost1 + cost0))/len(y) #-(1/m)[ylog(y')+(1-y)log(1-y')]
            return cost
        cost_list = np.zeros(iter,)
        for i in range(iter):
            weights = weights - alpha*dot(X.T,self.sigmoid(dot(X,weights))-np.reshape(y,(len(y),1))) #wnew=wold-alpha(y'-y)
            cost_list[i] = cost(weights)
            print("epoch",i,":",cost_list[i])

        self.weights = weights
        return cost_list 
    def predict(self,X):
        z = dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis


# In[14]:


obj1 = logistic_regression()
model= obj1.fit(x_train_array,y_train_array)

y_train_pred = obj1.predict(x_train_array)
y_test_pred = obj1.predict(x_test_array)


# In[15]:


train_acc=accuracy_score(y_train,y_train_pred)*100
train_acc


# In[16]:


test_acc=accuracy_score(y_test,y_test_pred)*100
test_acc

