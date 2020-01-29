#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')
import krylov.utils.loader as loader
import krylov.preconditioning
from krylov.methods import Methods


# In[2]:


# length = int(sys.argv[1])
length = 1081
version = 'EFG'
directory = 'data'


# In[3]:


A = loader.matrixLoader(directory,version,length)
b = loader.vectorLoader(directory,version,length)


# In[4]:


k = 8


# In[5]:


for k in range(1,10):
#     kskipmrr = Methods(A,b)
#     kskipmrr.kskipmrr(k=k)
    adaptivekskipmrr = Methods(A,b)
    adaptivekskipmrr.adaptivekskipmrr(k=k)
    variable = Methods(A,b)
    variable.variablekskipmrr(k=k)
#     Methods.multiplot([kskipmrr,adaptivekskipmrr],figsize=(12,8))
    Methods.multiplot([variable,adaptivekskipmrr],figsize=(12,8))    


# In[7]:


kskipjson = Methods(A,b)
adaptivejson = Methods(A,b)
kskipjson.json2instance('kskip.json')
adaptivejson.json2instance('adaptive.json')
Methods.multiplot([kskipmrr,adaptivekskipmrr],figsize=(12,8))


# In[ ]:




