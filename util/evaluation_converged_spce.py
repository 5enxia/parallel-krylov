#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append('../module/')
from krylov.methods import Methods,plt
import glob


# In[3]:


paths = glob.glob('1081/EFG/ada*')
ada_paths = sorted(paths)
paths = glob.glob('1081/EFG/kskip*')
kskip_paths = sorted(paths)


# In[4]:


import numpy as np
kskip_list = list()
adaptive_list = list()
for k in range(len(paths)):
    adaptive = Methods(A=np.zeros(1),b=np.zeros(1))
    kskip = Methods(A=np.zeros(1),b=np.zeros(1))
    adaptive.json2instance(ada_paths[k])
    kskip.json2instance(kskip_paths[k])
#     Methods.multiplot([adaptive,kskip],figsize=(12,8))
    if kskip.converged:
        kskip_list.append(str(kskip.solution_updates[kskip.iter-1]))
    else:
        kskip_list.append('発散')
    if adaptive.converged:
        i = 1
        while True:
            tmp = adaptive.solution_updates[adaptive.iter-i]
            if tmp is not 0:
                break
            i += 1
        adaptive_list.append(tmp)
    else:
        adaptive_list.append('発散')


# In[5]:


import pandas as pd


# In[6]:


df = pd.DataFrame(
    {
        'k':[k for k in range(1,31)],
        'k-skip':kskip_list,
        'adaptive':adaptive_list
    }
).set_index('k')
df


# In[7]:


df.to_csv('table.csv')


# In[ ]:




