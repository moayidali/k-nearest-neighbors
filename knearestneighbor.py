#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install pgmpy')


# In[17]:


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


# In[19]:


np.set_printoptions(precision=4,suppress=True)
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize']=7,4
plt.style.use('seaborn-whitegrid')



# In[111]:


address='C:/Users/HELLO/Documents/Electricdata.csv'
data=pd.read_csv(address)
data.columns=['tau1','tau2','tau3','tau4','p1','p2','p3','p4','g1','g2','g3','g4','stabf']
X_prime= data.ix[:,(0,1,2,3,4,5,6,7,8,9,10,11)].values
y = data.ix[:,12].values




# In[115]:


X=preprocessing.scale(X_prime)
print(X.shape)
print(y.shape)



# In[116]:


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.30, random_state=123)


# In[137]:



clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
print(clf)


# In[138]:


y_expect = y_test
y_pred = clf.predict(X_test)
print(metrics.classification_report(y_expect,y_pred))


# In[ ]:
print("Accuracy:",metrics.accuracy_score(y_expect, y_pred))



