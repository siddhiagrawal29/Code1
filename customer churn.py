# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 00:53:36 2023

@author: Siddhi Agrawal
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("Bank data.csv")
print(df.head())
df.shape
df.info()
df.isnull().sum()
df.describe()
for i in df.columns:
  dis = len(df[i].unique())
  print(f"{i} - {dis}")
df['churn'].value_counts()  
df.drop(columns=['customer_id'],inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df1=df.drop(['country','gender'],axis=1)
df1

fig, axes = plt.subplots(nrows = 4, ncols = 2)    
axes = axes.flatten()         
fig.set_size_inches(20,20)

for ax, col in zip(axes, df1.columns):
  sns.distplot(df[col], ax = ax)
  
fig, axes = plt.subplots(nrows = 4, ncols = 2)    
axes = axes.flatten()         
fig.set_size_inches(20,30)

for ax, col in zip(axes, df1.columns):
  sns.boxplot(y=df1[col], hue='churn', ax = ax , data=df1)
  ax.set_title(col)

fig, axes = plt.subplots(nrows = 4, ncols = 2)    
axes = axes.flatten()         
fig.set_size_inches(20,30)

for ax, col in zip(axes, df1.columns):
  sns.kdeplot(df1[col], ax = ax)
  
df = pd.get_dummies(df, columns=['country', 'gender'], drop_first=True)

correlation_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', linewidths=0.5, fmt='.2f')  

sns.scatterplot(x='credit_score', y='age', hue='churn', data=df)
sns.scatterplot(x='balance', y='age', hue='churn', data=df)
sns.scatterplot(x='estimated_salary', y='balance', hue='churn', data=df)  
X=df.drop(['churn'],axis=1)
y=df['churn']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None , 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
rfc=RandomForestClassifier()
rf_Grid=GridSearchCV(estimator=rfc,param_grid=param_grid,cv=3,verbose=0,n_jobs=-1,return_train_score=False)
rf_Grid.fit(X_train,y_train)
rf_Grid.best_params_

rf=RandomForestClassifier(**rf_Grid.best_params_)
rf.fit(X_train,y_train)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score , f1_score
import sklearn.metrics as metrics
y_pred1=rf.predict(X_test)
score_rf=accuracy_score(y_test,y_pred1)
score_rf
f1_rf=f1_score(y_pred1,y_test)
f1_rf
cm = metrics.confusion_matrix(y_test, y_pred1)
print(classification_report(y_test, y_pred1))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')

from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression()
param_grid = {
    'penalty': ['l1','l2'],
    'solver' :['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky'],
    'max_iter': [1000,1500,2000],
    'multi_class' :['auto', 'ovr', 'multinomial'],
    'class_weight' :['dict','balanced']
}
lgr_grid=GridSearchCV(estimator=lgr,param_grid=param_grid,cv=3,verbose=0,n_jobs=-1,return_train_score=True)
lgr_grid.fit(X_train,y_train)
lgr_grid.best_params_

log=LogisticRegression(**lgr_grid.best_params_)
log.fit(X_train,y_train)
y_pred2=log.predict(X_test)
f1_log=f1_score(y_pred2,y_test)
f1_log
score_log=accuracy_score(y_test,y_pred2)
score_log
cm = metrics.confusion_matrix(y_test, y_pred2)
print(classification_report(y_test, y_pred2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')

from sklearn.neighbors import KNeighborsClassifier 
param_grid = [{
    'n_neighbors': range(3,21),
    'algorithm':['ball_tree','kd_tree','brute'],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}]
knn_Grid= GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=3, verbose=0, n_jobs=-1)
knn_Grid.fit(X_train,y_train)
knn_Grid.best_params_
knn=KNeighborsClassifier(**knn_Grid.best_params_)
knn.fit(X_train,y_train)

y_pred3=knn.predict(X_test)
score_knn=accuracy_score(y_pred3,y_test)
score_knn
f1_knn=f1_score(y_pred3,y_test)
f1_knn
cm = metrics.confusion_matrix(y_test, y_pred3)
print(classification_report(y_test, y_pred3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')