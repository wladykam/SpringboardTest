#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:17:58 2020

@author: mattkelsey
"""
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option("max_columns",50)
pd.set_option("max_rows",30)
# json_ultimate = pd.read_json('C://Users/mattkelsey/Documents/ultimate_data_challenge.json', orient='records', lines=True)
# json_ultimate = pd.read_json('ultimate_data_challenge.json')
with open('ultimate_data_challenge.json') as f:
    data =  json.load(f)

user_data = pd.DataFrame(data)
mean_rating_bydriver = np.mean(user_data['avg_rating_by_driver'])
mean_rating_ofdriver = np.mean(user_data['avg_rating_of_driver'])
user_data.avg_rating_by_driver = user_data.avg_rating_by_driver.replace(np.nan,mean_rating_bydriver,regex=True)
user_data.avg_rating_of_driver = user_data.avg_rating_of_driver.replace(np.nan,mean_rating_ofdriver,regex=True)
user_data.phone.fillna(method='ffill',inplace=True)
user_data['active'] = user_data.last_trip_date > '2014-06-30'

# plot of users by city
user_data.city.value_counts(normalize=True).plot(kind='bar')
plt.show()

# plot of % of users with iPhone and Android
user_data.phone.value_counts(normalize=True).plot(kind='bar')
plt.show()

# plot of % of ultimate black users
user_data.ultimate_black_user.value_counts(normalize=True).plot(kind='bar')
plt.show()

# plot of % of active users
user_data.active.value_counts(normalize=True).plot(kind='bar')
plt.show()

# histogram of avg_dist
plt.hist(user_data.avg_dist,alpha=0.5,bins=30,range=[0,30])
plt.show()

# histogram of avg_rating_by_driver
plt.hist(user_data.avg_rating_by_driver,alpha=0.5,bins=20,range=[2.5,5])
plt.show()

# histogram of avg_rating_of_driver
plt.hist(user_data.avg_rating_of_driver,alpha=0.5,bins=20,range=[2.5,5])
plt.show()

# histogram of avg_surge
plt.hist(user_data.avg_surge,alpha=0.5,bins=10,range=[1,2])
plt.show()

# histogram of surge_pct
plt.hist(user_data.surge_pct,alpha=0.5,bins=20)
plt.show()

# histogram of trips_in_first_30_days
plt.hist(user_data.trips_in_first_30_days,alpha=0.5,bins=30,range=[0,25])
plt.show()

# histogram of weekday_pct
plt.hist(user_data.weekday_pct,alpha=0.5,bins=20)
plt.show()

plt.hist(user_data.last_trip_date,alpha=0.5,bins=20)
plt.show()

from datetime import datetime
user_data['last_trip_date'] = pd.to_datetime(user_data['last_trip_date'])
last_date = datetime(2014,7,1)
user_data['days_since_last_trip'] = (last_date - user_data['last_trip_date']).dt.days

user_data.drop(['last_trip_date','signup_date'],axis=1,inplace=True)
user_data = pd.get_dummies(user_data)
from sklearn.preprocessing import LabelEncoder

le_ub = LabelEncoder()
user_data.ultimate_black_user = le_ub.fit_transform(user_data.ultimate_black_user)

le_active = LabelEncoder()
user_data.active = le_active.fit_transform(user_data.active)
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# le_city = LabelEncoder()
# int_city = le_city.fit_transform(user_data.city)
# ohe_city = OneHotEncoder(sparse=False)
# int_city = int_city.reshape(len(int_city),1)
# ohe_city_fit = ohe_city.fit_transform(int_city)
# ohe_city_df = pd.DataFrame(ohe_city_fit)
# ohe_city_df.drop(2,axis=1,inplace=True)
# inverted_city = le_city.inverse_transform([np.argmax(ohe_city_fit[0,:])])

# le_phone = LabelEncoder()
# int_phone = le_phone.fit_transform(user_data.phone)
# ohe_phone = OneHotEncoder(sparse=False)
# int_phone = int_phone.reshape(len(int_phone),1)
# ohe_phone_fit = ohe_phone.fit_transform(int_phone)
# ohe_phone_df = pd.DataFrame(ohe_phone_fit)
# ohe_phone_df.drop(1,axis=1,inplace=True)
# inverted_phone = le_phone.inverse_transform([np.argmax(ohe_phone_fit[0,:])])

# le_ub = LabelEncoder()
# int_ub = le_ub.fit_transform(user_data.ultimate_black_user)
# ohe_ub = OneHotEncoder(sparse=False)
# int_ub = int_ub.reshape(len(int_ub),1)
# ohe_ub_fit = ohe_ub.fit_transform(int_ub)
# ohe_ub_df = pd.DataFrame(ohe_ub_fit)
# ohe_ub_df.drop(1,axis=1,inplace=True)
# inverted_ub = le_ub.inverse_transform([np.argmax(ohe_ub_fit[0,:])])

# le_active = LabelEncoder()
# int_active = le_active.fit_transform(user_data.active)
# ohe_active = OneHotEncoder(sparse=False)
# int_active = int_active.reshape(len(int_ub),1)
# ohe_active_fit = ohe_active.fit_transform(int_ub)
# ohe_active_df = pd.DataFrame(ohe_active_fit)
# ohe_active_df.drop(1,axis=1,inplace=True)
# inverted_active = le_active.inverse_transform([np.argmax(ohe_active_fit[0,:])])

import seaborn as sns
corr = user_data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# set up X and y arrays to be used in sklearn ML algorithms
X = user_data.drop(['active','city_Winterfell','phone_iPhone'],axis=1)
y = user_data['active'].values
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=42,stratify=y)
cv = RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=21)

def fit_model(X_train, y_train, model):
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

def model_run_all(X_train,y_train,y_test,model,model_name='Model'):
    '''Combines ML process into one function by using calls to fit_model(), roc_plot(), pr_plot()
    and model_stats() to provide cohesive ML model analysis'''
    y_pred, y_pred_prob = fit_model(X_train,y_train,model)
    roc_plot(y_test,y_pred_prob,title="{} ROC Curve".format(model_name))
    pr_plot(y_test,y_pred_prob,title="{} Precision Recall Curve".format(model_name))
    model_stats(y_test, y_pred, y_pred_prob,model=model_name)
    return y_pred, y_pred_prob

ada = AdaBoostClassifier()
ada_pred, ada_pred_prob = model_run_all(X_train, y_train, y_test, ada, model_name='AdaBoost Classifier')

log_reg = LogisticRegression()
log_reg_pred, log_reg_pred_prob = model_run_all(X_train, y_train, y_test, log_reg, model_name='Logistic Regression')


