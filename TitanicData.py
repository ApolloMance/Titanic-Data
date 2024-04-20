#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import sklearn


# ### Had to move file to correct directory

# In[6]:


import os
print(os.getcwd())


# ## Sample of Data Shown

# In[7]:


data = pd.read_csv('train.csv')
data.head(10)


# In[8]:


print('# of passengers:' + str(len(data.index)))


# # Analyzing Data

# In[9]:


sns.countplot(x='Survived',data = data)


# In[10]:


sns.countplot(x= 'Sex',data = data)


# In[11]:


sns.countplot(x= 'Survived', hue='Sex',data = data)


# In[14]:


sns.countplot(x='Survived', hue='Pclass', data=data)
plt.legend(title='Passenger Class')
plt.show()


# In[15]:


sns.countplot(x= 'Pclass',data = data)


# In[16]:


data['Age'].plot.hist()


# # Available Info

# In[17]:


data.info()


# # Data Cleaning

# In[18]:


data.isnull()


# In[19]:


data.isnull().sum()


# In[21]:


sns.heatmap(data.isnull(), yticklabels=False)


# ## Removing Mostly null Column

# In[22]:


data.drop('Cabin', axis=1, inplace= True)


# In[23]:


data.head(5)


# ## Removing 'NA' Values

# In[24]:


data.dropna(inplace=True)


# In[25]:


sns.heatmap(data.isnull(), yticklabels=False)


# In[26]:


data.isnull().sum()


# # Converting to Categorical Values

# In[27]:


data.head(2)


# In[39]:


sex= pd.get_dummies(data['Sex'], drop_first=True).astype(int)
sex.head(5)


# In[42]:


embark=pd.get_dummies(data['Embarked'], drop_first=True).astype(int)
embark.head(5)


# In[43]:


Pclass=pd.get_dummies(data['Pclass'], drop_first=True).astype(int)
Pclass.head(5)


# In[44]:


data=pd.concat([data,sex,embark,Pclass], axis=1)
data.head(5)


# In[45]:


data.drop(['Sex', 'Embarked','PassengerId','Name', 'Ticket', 'Pclass'],axis=1, inplace=True)
data.head(5)


# # Train Data

# In[53]:


X= data.drop('Survived', axis=1)
y= data['Survived']


# In[49]:


from sklearn.model_selection import train_test_split


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[51]:


from sklearn.linear_model import LogisticRegression


# In[63]:


logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train.values, y_train.values)
predictions = logmodel.predict(X_test.values)
print(predictions)


# In[66]:


from sklearn.metrics import classification_report
classification_report(y_test.values, predictions)


# In[67]:


from sklearn.metrics import accuracy_score


# In[68]:


accuracy_score(y_test.values, predictions)


# In[ ]:




