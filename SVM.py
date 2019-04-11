#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import scipy
import matplotlib.pyplot as plt
from pylab import rcParams
import urllib
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split

from sklearn import metrics


# In[2]:


address='C:/Users/HELLO/Documents/Electricdata.csv'
data=pd.read_csv(address)


# In[3]:


data.head()


# In[4]:


X =data.iloc[:,0:12]
y = data.iloc[:,12]
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.30, shuffle=True)


# In[7]:


y_train=[-1 if y==0 else 1 for y in y_train ]
y_test=[-1 if y==0 else 1 for  y in y_test ]


# In[8]:


y_train


# In[14]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[15]:


X_train =scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[17]:


from sklearn import svm
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=20)


# In[18]:


svm.fit(X_train , y_train)


# In[25]:


y_pred =svm.predict(X_test)


# In[22]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[26]:


test_score= accuracy_score(y_test,y_pred)


# In[27]:


test_score


# In[28]:


confusion_matrix(y_test, y_pred)


# In[30]:


from collections import Counter


# In[31]:


Counter(y_test)


# In[32]:


y_train_pred=svm.predict(X_train)


# In[33]:


accuracy_score(y_train,y_train_pred)


# In[ ]:




