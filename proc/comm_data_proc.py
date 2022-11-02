#!/usr/bin/env python
# coding: utf-8

# In[88]:


import numpy as np
import pandas as pd
import sys


# In[89]:


input_data_file = sys.argv[1]
output_data_file = sys.argv[2]
#data = pd.read_csv('./scjgw_ssztjbxx_test_202203041523.csv')
data = pd.read_csv(input_data_file)


# ### data

# In[19]:


selectedSchemas = [ 'RECORDID','JYFW', 'QYMC', 'JYCS', 'FDDBR', 'UPDATETIME']
mappedSchemas = {'JYFW': 'scope', 'RECORDID': 'entity_id', 'QYMC': 'name', 'JYCS': 'address', 'FDDBR': 'owner', 'UPDATETIME': 'timestamp'}


# In[20]:


data = data[selectedSchemas]

# In[21]:


# In[29]:


from datetime import datetime as dt
#t_object = dt.strptime(data['UPDATETIME'].loc[0], "%Y-%m-%d %H:%M:%S")


# In[32]:


def data2num(t_object):
    return t_object.year * 1e10 + t_object.month * 1e8 + t_object.day * 1e6 + t_object.hour * 1e4 + t_object.minute * 1e2 + t_object.second


# In[34]:


dataNum = []
for e in data['UPDATETIME']:
    #t_object = dt.strptime(e, "%Y-%m-%d %H:%M:%S")
    t_object = dt.strptime(e, "%d/%m/%Y %H:%M:%S")
    dataNum.append(data2num(t_object))


# In[35]:


dataNum[:10]


# In[38]:


data['UPDATETIME'] = np.array(dataNum, dtype='float')


# In[39]:


data


# In[33]:


data2num(t_object)


# In[26]:


for d in data.groupby('RECORDID'):
    print(d)
    break


# In[41]:


data.values[0]


# In[78]:


def processOne(cluster, eid, tid):
    #c = np.unique(cluster, axis=0)
    c = sorted(cluster, key=lambda x : x[-1])
    timestamp = 0
    c_ = [[tid, eid, tid] + list(c[0])[1:]]
    c_[0][-1] = timestamp
    tid += 1
    timestamp += 1
    for i in range(1, len(c)):
        if str(c[i-1]) == str(c[i]):
            continue
        if c[i-1][-1] == c[i][-1]:
            continue
        c_.append([tid, eid, tid] + list(c[i])[1:])
        c_[-1][-1] = timestamp
        tid += 1
        timestamp += 1
    return c_, tid


# In[79]:


from collections import defaultdict

dataG = defaultdict(list)
for record in data.values:
    dataG[record[0]].append(record)


# In[80]:


eid, tid = 0, 0
data_final = []
for k, v in dataG.items():
    #print(v)
    c_, tid = processOne(np.array(v), eid, tid)
    if len(c_) <= 1:
        tid -= len(c_)
        continue
    data_final += c_
    eid += 1


# In[81]:


#selectedSchemas = [ 'RECORDID','JYFW', 'QYMC', 'JYCS', 'FDDBR', 'UPDATETIME']
#mappedSchemas = {'JYFW': 'scope', 'RECORDID': 'entity_id', 'QYMC': 'name', 'JYCS': 'address', 'FDDBR': 'owner', 'UPDATETIME': 'timestamp'}

schema = ['id', 'entity_id', 'row_id'] + [mappedSchemas[s] for s in selectedSchemas if s != 'RECORDID']


# In[85]:


data_final_pd = pd.DataFrame(data_final, columns=schema)


# In[86]:


# In[ ]:


data_final_pd.to_csv(output_data_file, index=False)

