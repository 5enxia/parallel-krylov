#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')
import krylov.utils.loader as loader
import krylov.preconditioning
from krylov.methods import Methods


# In[2]:


length = int(sys.argv[1])
length = 1081
version = 'EFG'
directory = 'data'


# In[3]:


A = loader.matrixLoader(directory,version,length)
b = loader.vectorLoader(directory,version,length)


# In[15]:


for k in range(1,2):
    kskipmrr = Methods(A,b)
    kskipmrr.kskipmrr(k=k)
    
    kskipmrr.output('./results/' + str(length) + '/' + version + '/' + 'kskipmrr_' + str(k) + '.json')
    
    adaptivekskipmrr = Methods(A,b)
    adaptivekskipmrr.adaptivekskipmrr(k=k)
    
    adaptivekskipmrr.output('results/' + str(length) + '/' + version + '/' + ' adaptivekskipmrr_' + str(k) + '.json')


# In[ ]:




