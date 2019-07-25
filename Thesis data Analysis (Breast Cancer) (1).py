#!/usr/bin/env python
# coding: utf-8

# In[226]:


### Loading Libraries...
import pandas as pd
import numpy as np

### Graphic libraries
import matplotlib.pyplot as plt
import seaborn as sns 

### Some Scikit-learn utils
from sklearn.model_selection import train_test_split

### Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, auc

### Models
from xgboost import XGBClassifier, plot_importance

import itertools
from itertools import chain
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score

import lime
from lime.lime_tabular import LimeTabularExplainer

import eli5
from eli5.sklearn import PermutationImportance

import shap
from shap import TreeExplainer,KernelExplainer,LinearExplainer
shap.initjs()


### Some cosmetics add-ons
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


conda install -c plotly plotly 


# In[3]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


# In[246]:


data = pd.read_csv('/Users/odianosenakhibi/Downloads/Thesis/data/data.csv')


# In[5]:


pip install PDPbox


# In[247]:


np.random.seed(123) #ensure reproducibility

pd.options.mode.chained_assignment = None  #hide any pandas warnings


# In[248]:


from pdpbox import pdp, info_plots #for partial plots


# In[249]:


import seaborn as sns


# In[250]:


from xgboost import XGBClassifier, plot_importance


# In[251]:


from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, auc


# In[252]:


data.describe()


# In[253]:


# removing the last column as it is empty

data = data.drop(['id', 'Unnamed: 32'], axis = 1)

print(data.columns)


# In[254]:


data.describe()


# In[255]:


data.isna().sum().sum()


# In[256]:


# Reassign target
data.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)


# In[257]:


# 2 datasets
M = data[(data['diagnosis'] != 0)]
B = data[(data['diagnosis'] == 0)]


# In[258]:


plt.hist(data['diagnosis'])
plt.title('Diagnosis (M=1 , B=0)')
plt.show()


# In[259]:


features_mean=list(data.columns[1:11])
# split dataframe into two based on diagnosis
dataM=data[data['diagnosis'] ==1]
dataB=data[data['diagnosis'] ==0]


# In[260]:


#Stack the data
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(data[features_mean[idx]]) - min(data[features_mean[idx]]))/50
    ax.hist([dataM[features_mean[idx]],dataB[features_mean[idx]]], bins=np.arange(min(data[features_mean[idx]]), max(data[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, normed = True, label=['M','B'],color=['r','g'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()


# In[261]:


target = 'diagnosis'
features_list = list(data.columns)
features_list.remove(target)


# In[262]:


#correlation
correlation = data.corr()
#tick labels
matrix_cols = correlation.columns.tolist()
#convert to array
corr_array  = np.array(correlation)


# In[263]:


# for visualizing correlations
#Plotting
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   xgap = 2,
                   ygap = 2,
                   colorscale='Viridis',
                   colorbar   = dict() ,
                  )
layout = go.Layout(dict(title = 'Correlation Matrix for variables',
                        autosize = False,
                        height  = 720,
                        width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                     ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9)),
                       )
                  )
fig = go.Figure(data = [trace],layout = layout)
py.iplot(fig)


# In[265]:


cat_cols = list(data.select_dtypes('object').columns)
class_dict = {}
for col in cat_cols:
    data = pd.concat([data.drop(col, axis=1), pd.get_dummies(data[col])], axis=1)
data.head()


# In[266]:


# building train/test datasets on a 70/30 ratio
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape


# In[275]:


ML_models = {}
model_index = ['LR','RF','NN']
model_sklearn = [LogisticRegression(solver='liblinear',random_state=0),
                 RandomForestClassifier(n_estimators=100,random_state=0),
                 MLPClassifier([100]*5,early_stopping=True,learning_rate='adaptive',random_state=0)]
model_summary = []
for name,model in zip(model_index,model_sklearn):
    ML_models[name] = model.fit(X_train,y_train)
    preds = model.predict(X_test)
    model_summary.append([name,f1_score(y_test,preds,average='weighted'),accuracy_score(y_test,preds),
                          roc_auc_score(y_test,model.predict_proba(X_test)[:,1])])
ML_models


# In[276]:


model_summary = pd.DataFrame(model_summary,columns=['Name','F1_score','Accuracy','AUC_ROC'])
model_summary = model_summary.reset_index()
display(model_summary)


# In[269]:


g=sns.regplot(data=model_summary, x="index", y="AUC_ROC", fit_reg=False,
               marker="o", color="black", scatter_kws={'s':500})
 
for i in range(0,model_summary.shape[0]):
     g.text(model_summary.loc[i,'index'], model_summary.loc[i,'AUC_ROC']+0.02, 
            model_summary.loc[i,'Name'], 
            horizontalalignment='center',verticalalignment='top', size='large', color='black')


# In[278]:


#initialization of a explainer from LIME
explainer = LimeTabularExplainer(X_train.values,
                                 mode='classification',
                                 feature_names=X_train.columns,
                                 class_names=['Malign','Benign'])


# In[279]:


exp = explainer.explain_instance(X_test.head(1).values[0],
                                 ML_models['LR'].predict_proba,
                                 num_features=X_train.shape[1])
exp.show_in_notebook(show_table=True, show_all=True)


# In[280]:


exp = explainer.explain_instance(X_test.head(1).values[0],
                                 ML_models['RF'].predict_proba,
                                 num_features=X_train.shape[1])
exp.show_in_notebook(show_table=True, show_all=False)


# In[281]:


exp = explainer.explain_instance(X_test.head(1).values[0],
                                 ML_models['NN'].predict_proba,
                                 num_features=X_train.shape[1])
exp.show_in_notebook(show_table=True, show_all=False)


# In[231]:


#Eli5
eli5.show_weights(ML_models['LR'], feature_names = list(X_test.columns))


# In[277]:


eli5.show_prediction(ML_models['LR'], X_test.head(1).values[0],feature_names=list(X_test.columns))


# In[233]:


exp = PermutationImportance(ML_models['LR'],
                            random_state = 0).fit(X_test, y_test)
eli5.show_weights(exp,feature_names=list(X_test.columns))


# In[234]:


eli5.show_weights(ML_models['RF'],feature_names=list(X_test.columns))


# In[235]:


eli5.show_prediction(ML_models['RF'], X_test.head(1).values[0],feature_names=list(X_test.columns))


# In[236]:


exp = PermutationImportance(ML_models['RF'],
                            random_state = 0).fit(X_test, y_test)
eli5.show_weights(exp,feature_names=list(X_test.columns))


# In[237]:


eli5.show_weights(ML_models['NN'])


# In[238]:


#SHAP
explainer = LinearExplainer(ML_models['LR'], X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_test.head(1).values)
shap.force_plot(explainer.expected_value,
                shap_values,
                X_test.head(1).values,
                feature_names=X_test.columns)


# In[239]:


shap_values = explainer.shap_values(X_test.head(250).values)
shap.force_plot(explainer.expected_value,
                shap_values,
                X_test.head(250).values,
                feature_names=X_test.columns)


# In[240]:


shap_values = explainer.shap_values(X_test.values)
spplot = shap.summary_plot(shap_values, X_test.values, feature_names=X_test.columns)


# In[242]:


top4_cols = ['area_worst','radius_worst','radius_mean','perimeter_worst']
for col in top4_cols:
    shap.dependence_plot(col, shap_values, X_test)


# In[243]:


explainer = TreeExplainer(ML_models['RF'])
shap_values = explainer.shap_values(X_test.head(1).values)
shap.force_plot(explainer.expected_value[1],
                shap_values[1],
                X_test.head(1).values,
                feature_names=X_test.columns)


# In[245]:


X_train_kmeans = shap.kmeans(X_train, 10)
explainer = KernelExplainer(ML_models['NN'].predict_proba,X_train_kmeans)
shap_values = explainer.shap_values(X_test.head(1).values)
shap.force_plot(explainer.expected_value[1],
                shap_values[1],
                X_test.head(1).values,
                feature_names=X_test.columns)


# In[ ]:




