#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:36:52 2020

@author: mattkelsey
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import sys

pd.set_option("max_columns",20)
pd.set_option("max_rows",30)

# Import files 
users_df = pd.read_csv('takehome_users.csv',encoding='latin-1')
user_engagement_df = pd.read_csv('takehome_user_engagement.csv')

# Data wrangling to handle missing values and create extra columns for identifying adopted users
users_df.invited_by_user_id = users_df.invited_by_user_id.replace(np.nan,0,regex=True)
user_engagement_df['time_stamp'] =  pd.to_datetime(user_engagement_df['time_stamp'])
user_engagement_df['7_days_ahead'] =  user_engagement_df['time_stamp'] + timedelta(7)
users_df['creation_time'] =  pd.to_datetime(users_df['creation_time'])
users_df['last_session_creation_time'] =  pd.to_datetime(users_df['last_session_creation_time'])
users_df['last_minus_create'] =  (users_df['last_session_creation_time'] - users_df['creation_time']).astype('timedelta64[D]')*(-1)

# For loop to look through the data frame by user_id and then identify if they are adopted
adopted_users = []
for user_id in user_engagement_df['user_id'].unique():
    temp_df = user_engagement_df[user_engagement_df['user_id'] == user_id].copy()
    temp_df = temp_df.reset_index()
    count = 0
    if len(temp_df) >= 3:
        for i in range(len(temp_df)-2):
            if temp_df.iloc[i]['7_days_ahead'] >= temp_df.iloc[i+2]['time_stamp']:
                count += 1
                break
    if count > 0:
        adopted_users.append(user_id)

# Creates a column of booleans identifying whether or not a user is adopted
users_df['adopted'] = users_df['object_id'].isin(adopted_users)

# Getting data in numeric form for ML model
from sklearn.preprocessing import LabelEncoder

le_adopted = LabelEncoder()
users_df.adopted = le_adopted.fit_transform(users_df.adopted)

# Creating X, y columns for ML model
users_df_ml = users_df.drop(['creation_time','last_session_creation_time','last_minus_create','adopted', 
                             'name', 'email', 'object_id'], axis=1)
users_df_ml = pd.get_dummies(users_df_ml)
X = users_df_ml
y = users_df['adopted'].values
y = y.astype(int)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler

def fit_model(X_train, y_train, X_test, model):
    '''Fits an ML model and then uses the .predict() method to generate predictions
    for the binary classification as well as associated probabilities and returns those 
    predictions in a tuple'''
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    return (y_pred, y_pred_prob)

def roc_plot(y_test,y_pred_prob,title='ROC Curve'):
    '''Plots the ROC curve associated with classification probabilities
    generated from a ML model'''
    fpr,tpr,thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot([0,1],[0,1],'k--',color='blue')
    plt.plot(fpr,tpr,label='ROC-AUC: %0.2f'%roc_auc_score(y_test, y_pred_prob))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()

def pr_plot(y_test,y_pred_prob,title='Precision Recall Curve'):
    '''Plots the ROC curve associated with classification probabilities
    generated from a ML model'''
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.plot([1,0],[0,1],'k--',color='red')
    plt.plot(recall,precision,label='PR AUC: %0.2f'%(auc(recall,precision)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.show()
    

def model_stats(y_test,y_pred,y_pred_prob, model='Model'):
    '''Outputs summary statistics useful for comparing different ML models and their
    relative performance. Also outputs associated confusion matrix and classification report'''
    print("Tuned {} ROC-AUC score: {:0.2f}".format(model,roc_auc_score(y_test, y_pred_prob)))
    print("Tuned {} Precision Recall AUC score: {:0.2f}".format(model,average_precision_score(y_test, y_pred_prob)))
    print("Accuracy of tuned {}: {:0.2f}".format(model,accuracy_score(y_test, y_pred)))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def model_run_all(X_train,y_train, X_test, y_test, model,model_name='Model'):
    '''Combines ML process into one function by using calls to fit_model(), roc_plot(), pr_plot()
    and model_stats() to provide cohesive ML model analysis'''
    y_pred, y_pred_prob = fit_model(X_train,y_train,X_test,model)
    roc_plot(y_test,y_pred_prob,title="{} ROC Curve".format(model_name))
    pr_plot(y_test,y_pred_prob,title="{} Precision Recall Curve".format(model_name))
    model_stats(y_test, y_pred, y_pred_prob,model=model_name)
    return y_pred, y_pred_prob

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,random_state=42,stratify=y)
cv = RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=21)

# rf = RandomForestClassifier()
# n_estimators = [int(x) for x in np.linspace(start=200, stop =2000, num =10)]
# max_features = ['auto','sqrt']
# max_depth = [int(x) for x in np.linspace(5,110,num=11)]
# max_depth.append(None)
# min_samples_split = [1, 2, 5, 10]
# min_samples_leaf = [1,2,4]
# bootstrap = [True, False]
# random_grid = {'n_estimators': n_estimators,'max_features':max_features,'max_depth':max_depth,
#                 'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf,
#                 'bootstrap':bootstrap}
# rf_cv = RandomizedSearchCV(rf, param_distributions=random_grid,n_iter=5,cv=5,verbose=2,
#                             random_state=42,n_jobs=-1,scoring='accuracy')
# rf_cv_pred, rf_cv_pred_prob = model_run_all(X_train, y_train, X_test, y_test, rf_cv, 'Tuned Random Forest')
# print("Tuned Random Forest Parameters: {}".format(rf_cv.best_params_))

# Tuned Random Forest Parameters: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'auto',
#                                  'max_depth': 110, 'bootstrap': False}

# steps_rf = [('scaler',preprocessing.StandardScaler()),('RandomForest',RandomForestClassifier(n_estimators=200, min_samples_split=10,
#                                                                                              min_samples_leaf=4,max_depth=110, 
#                                                                                              max_features='auto',bootstrap=False,
#                                                                                              random_state=42))]
# pipeline_rf = Pipeline(steps_rf)

rf_best_cv = RandomForestClassifier(n_estimators=200, min_samples_split=10,min_samples_leaf=4,max_depth=110,
                                    max_features='auto',bootstrap=False,random_state=42)
rf_pred, rf_pred_prob = model_run_all(X_train, y_train, X_test, y_test, pipeline_rf,'Tuned RUS Random Forest')

# extracting feature importances from RF model and storing in DataFrame for plotting
importances = rf_best_cv.feature_importances_
importances = pd.Series(importances)
feature_list = list(X.columns)
feature_list = pd.Series(feature_list)
feature_importance_df = pd.concat([feature_list, importances],axis=1)
feature_importance_df.columns = ['features','importance']
feature_importance_df = feature_importance_df.sort_values(by='importance',ascending=False)

# plotting feature importances with corresponding labels in bar chart, descending order of importance
plt.bar(list(range((len(importances)))), feature_importance_df['importance'], orientation='vertical')
plt.xticks(list(range((len(importances)))), feature_importance_df['features'], rotation='vertical')
plt.ylabel('Feature Importance')
plt.xlabel('Variable')
plt.title('Variable Importance - Random Forest')
plt.show()