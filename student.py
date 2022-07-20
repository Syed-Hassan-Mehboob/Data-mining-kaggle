#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('student_data.csv')
df.head()


# In[3]:


df.isna().sum()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.sample(1).T


# In[7]:


bar_plot = sns.barplot(data=df,y='G3',x='schoolsup',hue='famsup');
bar_plot.set_title('The impact of school support and family support on G3 grade');


# In[8]:


bar_plot = sns.barplot(data=df,y='G3',x='guardian');
bar_plot.set_title('The impact of guardian on G3 grade');


# In[9]:


df.plot(kind='hist',y='G1',sharex=True)
df.plot(kind='hist',y='G2',sharex=True)
df.plot(kind='hist',y='G3',sharex=True);


# In[10]:


# Yes and No answers 
['schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'] # for one hot encoder


# In[11]:


t = df.reason.value_counts()
t.plot(kind='bar',x=t.index,y=t.values,title='The reasons for learning');


# In[12]:


t = df.traveltime.value_counts()
t.plot(kind='bar',x=t.index,y=t.values,title='The reasons for learning');


# In[13]:


df.info()


# In[14]:


X = df.drop('G3',axis=1)
y = df.G3


# In[15]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape


# In[16]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import SGDClassifier

onehot = OneHotEncoder()
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
numerical_cols = [cname for cname in X_train.columns if 
                X_train[cname].dtype in ['int64', 'float64']]
# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, object_cols),
        ('scaler', StandardScaler(),numerical_cols)
    ])

model = SGDClassifier(random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

