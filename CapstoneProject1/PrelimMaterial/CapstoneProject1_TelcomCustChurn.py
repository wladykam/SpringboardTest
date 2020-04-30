#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:23:28 2020

@author: mattkelsey
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

telcom_data = pd.read_csv('telcom_data.csv')

telcom_data.TotalCharges = telcom_data.TotalCharges.replace(' ',np.nan,regex=True)
telcom_data = telcom_data.dropna()
telcom_data = telcom_data.reset_index()
telcom_data = telcom_data.drop('index',axis=1)

telcom_data.customerID = telcom_data.customerID.astype('str')
telcom_data.gender = telcom_data.gender.astype('category')
telcom_data.SeniorCitizen = telcom_data.SeniorCitizen.astype('category')
telcom_data.Partner = telcom_data.Partner.astype('category')
telcom_data.Dependents = telcom_data.Dependents.astype('category')
telcom_data.tenure = telcom_data.tenure.astype('int')
telcom_data.PhoneService = telcom_data.PhoneService.astype('category')
telcom_data.MultipleLines = telcom_data.MultipleLines.astype('category')
telcom_data.InternetService = telcom_data.InternetService.astype('category')
telcom_data.OnlineSecurity = telcom_data.OnlineSecurity.astype('category')
telcom_data.OnlineBackup = telcom_data.OnlineBackup.astype('category')
telcom_data.DeviceProtection = telcom_data.DeviceProtection.astype('category')
telcom_data.TechSupport = telcom_data.TechSupport.astype('category')
telcom_data.StreamingTV = telcom_data.StreamingTV.astype('category')
telcom_data.StreamingMovies = telcom_data.StreamingMovies.astype('category')
telcom_data.Contract = telcom_data.Contract.astype('category')
telcom_data.PaperlessBilling = telcom_data.PaperlessBilling.astype('category')
telcom_data.PaymentMethod = telcom_data.PaymentMethod.astype('category')
telcom_data.TotalCharges = pd.to_numeric(telcom_data.TotalCharges)
telcom_data.Churn = telcom_data.Churn.astype('category')

plt.scatter(telcom_data.tenure, telcom_data.TotalCharges)
plt.xlabel('Tenure (months)')
plt.ylabel('Total Charges ($)')
plt.title ('Customer Lifetime Value over Time')
plt.show()

plt.hist(telcom_data.MonthlyCharges,bins=30)
plt.show()
plt.hist(telcom_data.TotalCharges,bins=30)
plt.show()
plt.hist(telcom_data.tenure,bins=10)
plt.show()
telcom_data.Contract.value_counts(normalize=True).plot(kind='bar')
plt.show()

contract_by_gender = telcom_data.groupby('gender').Contract.value_counts(normalize=True)
contract_by_gender.unstack().plot(kind='bar',stacked=True)
plt.show()

monthly_charge_month = telcom_data.MonthlyCharges[telcom_data.Contract == 'Month-to-month']
monthly_charge_year = telcom_data.MonthlyCharges[telcom_data.Contract == 'One year']
monthly_charge_two_year = telcom_data.MonthlyCharges[telcom_data.Contract == 'Two year']

plt.hist(monthly_charge_month,alpha=0.5,bins=30,label='month-to-month')
plt.hist(monthly_charge_year,alpha=0.5,bins=30,label='one year')
plt.hist(monthly_charge_two_year,alpha=0.5,bins=30,label='two year')
plt.legend()
plt.show()

total_charge_month = telcom_data.TotalCharges[telcom_data.Contract == 'Month-to-month']
total_charge_year = telcom_data.TotalCharges[telcom_data.Contract == 'One year']
total_charge_two_year = telcom_data.TotalCharges[telcom_data.Contract == 'Two year']

plt.hist(total_charge_month,alpha=0.5,bins=30,label='month-to-month')
plt.hist(total_charge_year,alpha=0.5,bins=30,label='one year')
plt.hist(total_charge_two_year,alpha=0.5,bins=30,label='two year')
plt.legend()
plt.show()

sns.violinplot(x='MonthlyCharges',y='Contract',hue='Churn',data=telcom_data)
plt.show()
sns.violinplot(x='TotalCharges',y='Contract',hue='Churn',data=telcom_data)
plt.show()

sns.boxplot(x='MonthlyCharges',y='Contract',hue='Churn',data=telcom_data)
plt.show()
sns.boxplot(x='TotalCharges',y='Contract',hue='Churn',data=telcom_data)
plt.show()

sns.regplot(x=telcom_data['MonthlyCharges'],y=telcom_data['TotalCharges'])
plt.show()


plt.violinplot(monthly_charge_month)
plt.show()
plt.violinplot(monthly_charge_year)
plt.show()
plt.violinplot(monthly_charge_two_year)
plt.show()

telcom_data.SeniorCitizen.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.Partner.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.Dependents.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.PhoneService.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.MultipleLines.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.InternetService.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.OnlineSecurity.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.OnlineBackup.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.DeviceProtection.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.TechSupport.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.StreamingTV.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.StreamingMovies.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.PaperlessBilling.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.PaymentMethod.value_counts(normalize=True).plot(kind='bar')
plt.show()
telcom_data.Churn.value_counts(normalize=True).plot(kind='bar')
plt.show()

churn_by_payment = telcom_data.groupby('PaymentMethod').Churn.value_counts(normalize=True)
churn_by_payment.unstack().plot(kind='bar',stacked=True)
plt.show()


churn_by_contract = telcom_data.groupby('Contract').Churn.value_counts(normalize=True)
churn_by_contract.unstack().plot(kind='bar',stacked=True)
plt.show()

churn_by_partner = telcom_data.groupby('Partner').Churn.value_counts(normalize=True)
churn_by_partner.unstack().plot(kind='bar',stacked=True)
plt.show()

churn_by_dependents = telcom_data.groupby('Dependents').Churn.value_counts(normalize=True)
churn_by_dependents.unstack().plot(kind='bar',stacked=True)
plt.show()

churn_by_senior = telcom_data.groupby('SeniorCitizen').Churn.value_counts(normalize=True)
churn_by_senior.unstack().plot(kind='bar',stacked=True)
plt.show()

churn_by_billing = telcom_data.groupby('PaperlessBilling').Churn.value_counts(normalize=True)
churn_by_billing.unstack().plot(kind='bar',stacked=True)
plt.show()

#Observed Survival Plots
tenure_group = telcom_data.groupby(['tenure','gender']).Churn.value_counts(normalize=True)
tenure_group_unstack = tenure_group.unstack().drop('No',axis=1)
tenure_group_unstack.plot(kind='line')
plt.show()
#plt.xticks(ticks=tenure_group_unstack.index)
tenure_group_double_unstack = tenure_group_unstack.unstack()
tenure_group_double_unstack.plot(kind='line')
plt.show()

payment_group = telcom_data.groupby(['tenure','PaymentMethod']).Churn.value_counts(normalize=True)
payment_group_unstack = payment_group.unstack().drop('Yes',axis=1)
payment_group_unstack.plot(kind='line')
plt.show()
payment_group_double_unstack = payment_group_unstack.unstack()
payment_group_double_unstack.plot(kind='line')
plt.show()

contract_group = telcom_data.groupby(['tenure','Contract']).Churn.value_counts(normalize=True)
contract_group_unstack = contract_group.unstack().drop('Yes',axis=1)
contract_group_unstack.plot(kind='line')
contract_group_double_unstack = contract_group_unstack.unstack()
contract_group_double_unstack.plot(kind='line')
plt.show()

from pysurvival.models.non_parametric import KaplanMeierModel
from pysurvival.utils.display import display_non_parametric
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


#T = np.round(np.abs(np.random.normal(10,10,1000)),1)
#E = np.random.binomial(1,0.3,1000)

# km_model = KaplanMeierModel()
# km_model.fit(T,E,alpha=0.95)
# display_non_parametric(km_model)

# def my_custom_parametric_plot(km_model, figure_size = (18, 5) ):
#     """ Plotting the survival function and its lower and upper bounds 
#         Parameters:
#         -----------
#         * km_model : pysurvival Non-Parametric model
#             The model that will be used for prediction
#         * figure_size: tuple of double (default= (18, 5))
#             width, height in inches representing the size of the chart 
#     """

#     # Check that the model is a Non-Parametric model
#     if 'kaplan' not in km_model.name.lower() :
#         error = "This function can only take as input a Non-Parametric model"
#         raise NotImplementedError(error)

#     # Title of the chart
#     if 'smooth' in km_model.name.lower() :
#         is_smoothed = True
#         title = 'Smooth Kaplan-Meier Survival function'
#     else:
#         is_smoothed = False
#         title = 'Kaplan-Meier Survival function'

#     # Initializing the chart
#     fig, ax = plt.subplots(figsize=figure_size )

#     # Extracting times and survival function
#     times, survival = km_model.times, km_model.survival

#     # Plotting Survival
#     plt.plot(times, survival, label = title, 
#              color = 'blue', lw = 3)  

#     # Defining the x-axis and y-axis
#     ax.set_xlabel('Time')
#     ax.set_ylabel( 'S(t) Survival function' )
#     ax.set_ylim([0.0, 1.05])
#     ax.set_xlim([0.0, max(times)*1.01])
#     vals = ax.get_yticks()
#     ax.set_yticklabels(['{:.1f}%'.format(v*100) for v in vals])
#     plt.title(title, fontsize=25)

#     # Extracting times and survival function
#     times, survival = km_model.times, km_model.survival

#     if is_smoothed :

#         # Display
#         plt.plot(times, survival, label = 'Original Kaplan-Meier', 
#                  color = '#f44141', ls = '-.', lw = 2.5)        
#         plt.legend(fontsize=15)
#         plt.show()

#     else:

#         # Extracting CI
#         survival_ci_upper = km_model.survival_ci_upper
#         survival_ci_lower = km_model.survival_ci_lower

#         # Plotting the Confidence Intervals
#         plt.plot(times, survival_ci_upper, 
#                  color='red', alpha =0.1, ls='--')
#         plt.plot(times, survival_ci_lower, 
#                  color='red', alpha =0.1, ls='--')

#         # Filling the areas between the Survival and Confidence Intervals curves
#         plt.fill_between(times, survival, survival_ci_lower, 
#                 label='Confidence Interval - lower', color='red', alpha =0.2)
#         plt.fill_between(times, survival, survival_ci_upper, 
#                 label='Confidence Interval - upper', color='red', alpha =0.2)
        
#         # Display
#         plt.legend(fontsize=15)
#         plt.show()


le_gender = LabelEncoder()
telcom_data.gender = le_gender.fit_transform(telcom_data.gender)
le_senior = LabelEncoder()
telcom_data.SeniorCitizen = le_senior.fit_transform(telcom_data.SeniorCitizen)
le_partner = LabelEncoder()
telcom_data.Partner = le_partner.fit_transform(telcom_data.Partner)
le_dependents = LabelEncoder()
telcom_data.Dependents = le_dependents.fit_transform(telcom_data.Dependents)
le_phone = LabelEncoder()
telcom_data.PhoneService = le_phone.fit_transform(telcom_data.PhoneService)
le_multi = LabelEncoder()
telcom_data.MultipleLines = le_multi.fit_transform(telcom_data.MultipleLines)
le_internet = LabelEncoder()
telcom_data.InternetService = le_internet.fit_transform(telcom_data.InternetService)
le_security = LabelEncoder()
telcom_data.OnlineSecurity = le_security.fit_transform(telcom_data.OnlineSecurity)
le_backup = LabelEncoder()
telcom_data.OnlineBackup = le_backup.fit_transform(telcom_data.OnlineBackup)
le_protection = LabelEncoder()
telcom_data.DeviceProtection = le_protection.fit_transform(telcom_data.DeviceProtection)
le_support = LabelEncoder()
telcom_data.TechSupport = le_support.fit_transform(telcom_data.TechSupport)
le_s_tv = LabelEncoder()
telcom_data.StreamingTV = le_s_tv.fit_transform(telcom_data.StreamingTV)
le_s_movie = LabelEncoder()
telcom_data.StreamingMovies = le_s_movie.fit_transform(telcom_data.StreamingMovies)
le_contract = LabelEncoder()
telcom_data.Contract = le_contract.fit_transform(telcom_data.Contract)
le_billing = LabelEncoder()
telcom_data.PaperlessBilling = le_billing.fit_transform(telcom_data.PaperlessBilling)
le_payment = LabelEncoder()
telcom_data.PaymentMethod = le_payment.fit_transform(telcom_data.PaymentMethod)
le_churn = LabelEncoder()
telcom_data.Churn = le_churn.fit_transform(telcom_data.Churn)

#telcom_data.Churn[telcom_data.Churn == 'Yes'] = 1
#telcom_data.Churn[telcom_data.Churn == 'No'] = 0

#telcom_data.gender[telcom_data.gender == 'Male'] = 1
#telcom_data.gender[telcom_data.gender == 'Female'] = 0


T_male = telcom_data[telcom_data.gender == 1].tenure
E_male = telcom_data[telcom_data.gender== 1].Churn

T_female = telcom_data[telcom_data.gender == 0].tenure
E_female = telcom_data[telcom_data.gender == 0].Churn

km_male_model = KaplanMeierModel()
km_male_model.fit(T_male, E_male, alpha=0.95)

km_female_model = KaplanMeierModel()
km_female_model.fit(T_female, E_female, alpha=0.95)

#display_non_parametric(km_male_model)
plt.plot(km_female_model.times, km_female_model.survival,label='Female')
plt.plot(km_male_model.times, km_male_model.survival,label='Male')
plt.xlabel('Tenure - Months')
plt.ylabel('Probability of Survival')
plt.title('Kaplan-Meier Survival by Gender')
plt.legend()
plt.show()

T_bank_transfer = telcom_data[telcom_data.PaymentMethod == 0].tenure
E_bank_transfer = telcom_data[telcom_data.PaymentMethod == 0].Churn
T_credit_card = telcom_data[telcom_data.PaymentMethod == 1].tenure
E_credit_card = telcom_data[telcom_data.PaymentMethod == 1].Churn
T_e_check = telcom_data[telcom_data.PaymentMethod == 2].tenure
E_e_check = telcom_data[telcom_data.PaymentMethod == 2].Churn
T_mail_check = telcom_data[telcom_data.PaymentMethod == 3].tenure
E_mail_check = telcom_data[telcom_data.PaymentMethod == 3].Churn

km_bank_transfer_model = KaplanMeierModel()
km_bank_transfer_model.fit(T_bank_transfer, E_bank_transfer)
km_credit_card_model = KaplanMeierModel()
km_credit_card_model.fit(T_credit_card, E_credit_card)
km_e_check_model = KaplanMeierModel()
km_e_check_model.fit(T_e_check, E_e_check)
km_mail_check_model = KaplanMeierModel()
km_mail_check_model.fit(T_mail_check, E_mail_check)

plt.plot(km_bank_transfer_model.times, km_bank_transfer_model.survival,label='Bank Transfer')
# plt.plot(km_bank_transfer_model.times, km_bank_transfer_model.survival_ci_upper)
# plt.plot(km_bank_transfer_model.times, km_bank_transfer_model.survival_ci_lower)
plt.plot(km_credit_card_model.times, km_credit_card_model.survival,label='Credit Card')
plt.plot(km_e_check_model.times, km_e_check_model.survival,label='Electronic Check')
plt.plot(km_mail_check_model.times, km_mail_check_model.survival,label='Mailed Check')
plt.xlabel('Tenure - Months')
plt.ylabel('Probability of Survival')
plt.title('Kaplan-Meier Survival by Payment Method')
plt.legend()
plt.show()

T_month = telcom_data[telcom_data.Contract == 0].tenure
E_month = telcom_data[telcom_data.Contract == 0].Churn
T_year = telcom_data[telcom_data.Contract == 1].tenure
E_year = telcom_data[telcom_data.Contract == 1].Churn
T_two_year = telcom_data[telcom_data.Contract == 2].tenure
E_two_year = telcom_data[telcom_data.Contract == 2].Churn

km_month_model = KaplanMeierModel()
km_month_model.fit(T_month, E_month)
km_year_model = KaplanMeierModel()
km_year_model.fit(T_year, E_year)
km_two_year_model = KaplanMeierModel()
km_two_year_model.fit(T_two_year, E_two_year)

plt.plot(km_month_model.times, km_month_model.survival,label='Month-to-Month')
plt.plot(km_year_model.times, km_year_model.survival,label='One Year')
plt.plot(km_two_year_model.times, km_two_year_model.survival,label='Two Year')
plt.xlabel('Tenure - Months')
plt.ylabel('Probability of Survival')
plt.title('Kaplan-Meier Survival by Payment Method')
plt.legend()
plt.show()




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
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.display import integrated_brier_score

#telcom_data.PaperlessBilling[telcom_data.PaperlessBilling == 'Yes'] = 1
#telcom_data.PaperlessBilling[telcom_data.PaperlessBilling == 'No'] = 0


X = telcom_data.drop(['Churn','customerID'],axis=1)
#X_scaled = preprocessing.scale(X)
y = telcom_data['Churn'].values
y = y.astype('int')

feature_list = list(X.columns)

def fit_model(X_train, y_train, model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    return (y_pred, y_pred_prob)

def roc_plot(y_test,y_pred_prob,title='ROC Curve'):
    fpr,tpr,thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot([0,1],[0,1],'k--',color='blue')
    plt.plot(fpr,tpr,label='ROC-AUC: %0.2f'%roc_auc_score(y_test, y_pred_prob))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()
    #return plt.plot(fpr,tpr,label='ROC-AUC: %0.2f'%roc_auc_score(y_test, y_pred_prob))

def pr_plot(y_test,y_pred_prob,title='Precision Recall Curve'):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.plot([1,0],[0,1],'k--',color='red')
    plt.plot(recall,precision,label='PR AUC: %0.2f'%(auc(recall,precision)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.show()
    
def model_stats(y_test,y_pred,y_pred_prob, model='Model'):
    print("Tuned {} ROC-AUC score: {:0.2f}".format(model,roc_auc_score(y_test, y_pred_prob)))
    print("Tuned {} Precision Recall AUC score: {:0.2f}".format(model,average_precision_score(y_test, y_pred_prob)))
    print("Accuracy of tuned {}: {:0.2f}".format(model,accuracy_score(y_test, y_pred)))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def model_run_all(X_train,y_train,y_test,model,model_name='Model'):
    y_pred, y_pred_prob = fit_model(X_train,y_train,model)
    roc_plot(y_test,y_pred_prob,title="{} ROC Curve".format(model_name))
    pr_plot(y_test,y_pred_prob,title="{} Precision Recall Curve".format(model_name))
    model_stats(y_test, y_pred, y_pred_prob,model=model_name)
    return y_pred, y_pred_prob


#logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=42,stratify=y)

#X_csf = telcom_data.drop(['customerID','tenure','Churn'],axis=1)
#T_csf = telcom_data['tenure'].values
#E_csf = telcom_data['Churn'].values
#index_train, index_test = train_test_split(range(len(telcom_data)),test_size=0.3,random_state=42,stratify=y)

X_rsf = X.drop('tenure',axis=1)
T_rsf = X['tenure'].values
E_rsf = y

index_train, index_test = X_train.index, X_test.index

X_rsf_train, X_rsf_test = X_rsf.loc[index_train,:], X_rsf.loc[index_test,:]
T_rsf_train, T_rsf_test = T_rsf[index_train], T_rsf[index_test]
E_rsf_train, E_rsf_test = E_rsf[index_train], E_rsf[index_test]

km_base_model = KaplanMeierModel()
km_base_model.fit(T_rsf_test, E_rsf_test)

cv = RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=21)

rsf = RandomSurvivalForestModel(num_trees=200)
#rsf_num_trees = [100,200,300,500]
#rsf_max_depth = [5,10,15,20,25,30,35,40,45,50]
#rsf_min_node_size = [3, 5, 7, 9]
#param_grid_rsf = {'num_trees':rsf_num_trees,'max_depth':rsf_max_depth,'min_node_size':rsf_min_node_size}
rsf.fit(X_rsf_train,T_rsf_train,E_rsf_train,max_features='sqrt',max_depth=36,min_node_size=4,seed=21)
#rsf_cv = RandomizedSearchCV(rsf, param_distributions=param_grid_rsf, cv=cv,scoring='accuracy',random_state=42,)
#rsf_cv.fit(X_rsf_train,T_rsf_train,E_rsf_train)
c_index = concordance_index(rsf,X_rsf_test,T_rsf_test,E_rsf_test)
print('C-index: {:0.2f}'.format(c_index))
ibs = integrated_brier_score(rsf, X_rsf_test, T_rsf_test, E_rsf_test)
print('IBS: {:0.2f}'.format(ibs))

# Initializing the figure
fig, ax = plt.subplots(figsize=(8, 4))

# Randomly extracting a data-point that experienced an event 
choices = np.argwhere((E_rsf_test==1.)&(T_rsf_test>=1)).flatten()
np.random.seed(16)
#k = np.random.choice( choices, 1)[0]

# Saving the time of event
t = T_rsf_test[choices]

# Computing the Survival function for all times t
#survival = rsf.predict_survival(X_rsf_test.values[t, :]).flatten()
survival = rsf.predict_survival(X_rsf_test.values)
survival_avg = survival.mean(axis=0)
#actual = km_base_model.predict_survival(X_csf_test.values[k, :]).flatten()

# Displaying the functions
plt.plot(rsf.times, survival_avg, color='blue', label='RSF Predicted', lw=2, ls = '-.')
plt.plot(km_base_model.times, km_base_model.survival,color='red',label='Actual',lw=1)
#plt.plot(sim.times, actual, color = 'red', label='actual', lw=2)
# Actual time
#plt.axvline(x=t, color='black', ls ='--')
#ax.annotate('T={:.1f}'.format(t), xy=(t, 0.5), xytext=(t, 0.5), fontsize=12)

# Show everything
title = "Comparing Survival functions between Actual and RSF Predicted"
plt.legend(fontsize=12)
plt.title(title, fontsize=15)
plt.ylim(0, 1.05)
plt.show()

#creating Logistic Regression model
c_space = [.0001,.001,.1,1,10,100]
param_grid = {'LogisticRegression__C':c_space}
steps_logreg = [('scaler',preprocessing.StandardScaler()),('LogisticRegression',LogisticRegression())]
pipeline_logreg = Pipeline(steps_logreg)
logreg_cv = GridSearchCV(pipeline_logreg, param_grid,cv=cv,scoring='accuracy')
y_logreg_pred, y_logreg_pred_prob = model_run_all(X_train, y_train, y_test, logreg_cv,model_name='Logistic Regression')

##creating k-NN model
n_neighbors = np.arange(1,21,2)
weights = ['uniform','distance']
metric = ['euclidean','manhattan','minkowski']

knn = KNeighborsClassifier()
param_grid_knn = {'n_neighbors':n_neighbors,'weights':weights,'metric':metric}
#cv_params = RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=21)
knn_cv = GridSearchCV(knn,param_grid=param_grid_knn,cv=cv,scoring='accuracy')
y_knn_pred, y_knn_pred_prob = model_run_all(X_train, y_train, y_test, knn_cv,model_name='kNN')
print("Tuned k-NN Parameters: {}".format(knn_cv.best_params_))
# knn_cv.fit(X_train,y_train)


# print("Tuned k-NN Parameters: {}".format(knn_cv.best_params_))
# print("Tuned k-NN Accuracy: {}".format(knn_cv.best_score_))

# # # n_estimators = [int(x) for x in np.linspace(start=200, stop =2000, num =10)]
# # # max_features = ['auto','sqrt']
# # # max_depth = [int(x) for x in np.linspace(5,110,num=11)]
# # # max_depth.append(None)
# # # min_samples_split = [1, 2, 5, 10]
# # # min_samples_leaf = [1,2,4]
# # # bootstrap = [True, False]
# # # random_grid = {'n_estimators': n_estimators,'max_features':max_features,'max_depth':max_depth,
# # #                'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf,
# # #                'bootstrap':bootstrap}

# # rf = RandomForestClassifier()
# # # rf_cv = RandomizedSearchCV(rf, param_distributions=random_grid,n_iter=5,cv=5,verbose=2,
# # #                            random_state=42,n_jobs=-1,scoring='accuracy')
# # #rf_cv.fit(X_train,y_train)
# # # print("Tuned Logistic Regression Parameter: {}".format(rf_cv.best_params_))
# # # #Tuned Logistic Regression Parameter: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 
# # # #'max_features': 'sqrt', 'max_depth': 36, 'bootstrap': True}
# # # print("Tuned Logistic Regression Accuracy: {}".format(rf_cv.best_score_))
# # # #Tuned Logistic Regression Accuracy: 0.7980541455160745
# # # y_rf_pred = rf_cv.predict(X_test)
# # # y_rf_pred_prob = rf_cv.predict_proba(X_test)[:,1]
# # # fpr_rf,tpr_rf,thresholds_rf = roc_curve(y_test, y_rf_pred_prob)
# # # plt.plot([0,1],[0,1],'k--',color='blue')
# # # plt.plot(fpr_rf,tpr_rf,label='ROC-AUC: %0.2f'%(roc_auc_score(y_test, y_rf_pred_prob)))
# # # plt.xlabel('False Positive Rate')
# # # plt.ylabel('True Positive Rate')
# # # plt.title('Random Forest ROC Curve')
# # # plt.legend()
# # # plt.show()

# # #n_estimators=[int(x) for x in np.linspace(start=50, stop = 300, num =6)]
# # #max_depth = [36]
# # #min_samples_split = [3, 4, 5, 6, 7]
# # #min_samples_leaf = [1,2,3,4]
# # #bootstrap=[True]
# # #random_grid = {'n_estimators': n_estimators,'max_depth':max_depth,
# # #                'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf,
# # #                'bootstrap':bootstrap}

# # # rf_gs_cv = GridSearchCV(rf, param_grid=random_grid,cv=cv)
# # # rf_gs_cv.fit(X_train,y_train)
# # # print("Tuned Logistic Regression Parameter: {}".format(rf_gs_cv.best_params_))
# # # #Tuned Logistic Regression Parameter: {'bootstrap': True, 'max_depth': 36, 'min_samples_leaf': 4, 'min_samples_split': 4,
# # # # 'n_estimators': 200}
# # # print("Tuned Logistic Regression Accuracy: {}".format(rf_gs_cv.best_score_))
# # # #Tuned Logistic Regression Accuracy: 0.8014359016686614
# # # y_gs_rf_pred = rf_gs_cv.predict(X_test)
# # # y_gs_rf_pred_prob = rf_gs_cv.predict_proba(X_test)[:,1]
# # # fpr_rf,tpr_rf,thresholds_rf = roc_curve(y_test, y_gs_rf_pred_prob)
# # # plt.plot([0,1],[0,1],'k--',color='blue')
# # # plt.plot(fpr_rf,tpr_rf,label='ROC-AUC: %0.2f'%(roc_auc_score(y_test, y_gs_rf_pred_prob)))
# # # plt.xlabel('False Positive Rate')
# # # plt.ylabel('True Positive Rate')
# # # plt.title('Random Forest ROC Curve')
# # # plt.legend()
# # # plt.show()

# n_estimators=[200]
# min_samples_split=[4]
# min_samples_leaf=[4]
# max_depth=[36]
# max_features=['sqrt']
# bootstrap=[True]
# rf = RandomForestClassifier()
# random_grid ={'n_estimators': n_estimators,'max_depth':max_depth,
#               'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf,
#               'bootstrap':bootstrap}

# rf_best_cv = GridSearchCV(rf, param_grid=random_grid,cv=cv,scoring='average_precision')
rf_best_cv = RandomForestClassifier(n_estimators=200, min_samples_split=4,min_samples_leaf=4,max_depth=36,
                                    max_features='sqrt',bootstrap=True)
y_rf_pred, y_rf_pred_prob = model_run_all(X_train, y_train, y_test, rf_best_cv,model_name='Random Forest')



##Feature Importance Analysis - as implied by Random Forest model
importances = rf_best_cv.feature_importances_
#importances = sorted(importances,reverse=True)
feature_importances = [(feature,round(importance,2)) for feature, importance in zip(feature_list,importances)]
feature_importances = sorted(feature_importances, key= lambda x:x[1],reverse=True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
importances = pd.Series(importances)
feature_list = pd.Series(feature_list)
feature_importance_df = pd.concat([feature_list, importances],axis=1)
feature_importance_df.columns = ['features','importance']
feature_importance_df = feature_importance_df.sort_values(by='importance',ascending=False)

plt.bar(list(range((len(importances)))), feature_importance_df['importance'], orientation='vertical')
#plt.yticks(list(range((len(importances)))), feature_importance_df['features'], rotation='horizontal')
plt.xticks(list(range((len(importances)))), feature_importance_df['features'], rotation='vertical')
plt.ylabel('Feature Importance')
plt.xlabel('Variable')
plt.title('Variable Importance - Random Forest')
plt.show()

##SVM Model
svm_steps = [('scaler',preprocessing.StandardScaler()),('SVM',SVC())]
svm_cs = [10]
gammas = [.01]
#Tuned Logistic Regression Parameter: {'SVM__C': 10, 'SVM__gamma': 0.01, 'SVM__probability': True}
#Tuned Logistic Regression Accuracy: 0.7976468160620692
probability = [True]
svm_param_grid = {'SVM__C':svm_cs, 'SVM__gamma':gammas, 'SVM__probability':probability}
svm_pipeline = Pipeline(svm_steps)
svm_cv = GridSearchCV(svm_pipeline, param_grid=svm_param_grid, cv=cv,scoring='accuracy')
y_svm_pred, y_svm_pred_prob = model_run_all(X_train, y_train, y_test, svm_cv,model_name='SVM')


##AdaBoost model
n_estimators_boost = [300]
learning_rate_boost = [.1]
param_grid_boost = {'n_estimators':n_estimators_boost, 'learning_rate':learning_rate_boost}
ada = AdaBoostClassifier()
ada_cv = GridSearchCV(ada, param_grid=param_grid_boost,cv=cv, scoring='accuracy')
y_ada_pred, y_ada_pred_prob =model_run_all(X_train, y_train, y_test, ada_cv,model_name='AdaBoost')

my_estimators = [('knn',knn_cv),('log_reg',logreg_cv),('rf',rf_best_cv),('svm',svm_cv),('ada',ada_cv)]
vc_model = VotingClassifier(my_estimators,voting='soft')
y_vc_pred, y_vc_pred_prob = model_run_all(X_train, y_train, y_test, vc_model,model_name='Voting Classifier')


# fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_logreg_pred_prob)
# fpr_knn, tpr_knn, _ = roc_curve(y_test, y_knn_pred_prob)
# fpr_rf, tpr_rf, _ = roc_curve(y_test, y_rf_pred_prob)
# fpr_svm, tpr_svm, _ = roc_curve(y_test, y_svm_pred_prob)
# fpr_ada, tpr_ada, _ = roc_curve(y_test, y_ada_pred_prob)
# fpr_vc, tpr_vc, _ = roc_curve(y_test, y_vc_pred_prob)
# plt.plot([0,1],[0,1],'k--',color='blue')
# plt.plot(fpr_logreg,tpr_logreg,label='LR')
# plt.plot(fpr_knn,tpr_knn,label='kNN')
# plt.plot(fpr_rf,tpr_rf,label='RF')
# plt.plot(fpr_svm,tpr_svm,label='SVM')
# plt.plot(fpr_ada,tpr_ada,label='ADA')
# plt.plot(fpr_vc,tpr_vc,label='VC')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves - All Models')
# plt.legend()
# plt.show()

# p_logreg, r_logreg, _ = precision_recall_curve(y_test, y_logreg_pred_prob)
# p_knn, r_knn, _ = precision_recall_curve(y_test, y_knn_pred_prob)
# p_rf, r_rf, _ = precision_recall_curve(y_test, y_rf_pred_prob)
# p_svm, r_svm, _ = precision_recall_curve(y_test, y_svm_pred_prob)
# p_ada, r_ada, _ = precision_recall_curve(y_test, y_ada_pred_prob)
# p_vc, r_vc, _ = precision_recall_curve(y_test, y_vc_pred_prob)
# plt.plot([1,0],[0,1],'k--',color='red')
# plt.plot(r_logreg,p_logreg,label='LR')
# plt.plot(r_knn,p_knn,label='kNN')
# plt.plot(r_rf,p_rf,label='RF')
# plt.plot(r_svm,p_svm,label='SVM')
# plt.plot(r_ada,p_ada,label='ADA')
# plt.plot(r_vc,p_vc,label='VC')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('PR Curves - All Models')
# plt.legend()
# plt.show()

plt.figure(figsize=[10,6])
plt.subplot(1,2,1)
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_logreg_pred_prob)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_knn_pred_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_rf_pred_prob)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_svm_pred_prob)
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_ada_pred_prob)
fpr_vc, tpr_vc, _ = roc_curve(y_test, y_vc_pred_prob)
plt.plot([0,1],[0,1],'k--',color='blue')
plt.plot(fpr_logreg,tpr_logreg,label='LR')
plt.plot(fpr_knn,tpr_knn,label='kNN')
plt.plot(fpr_rf,tpr_rf,label='RF')
plt.plot(fpr_svm,tpr_svm,label='SVM')
plt.plot(fpr_ada,tpr_ada,label='ADA')
plt.plot(fpr_vc,tpr_vc,label='VC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend()

plt.subplot(1,2,2)
p_logreg, r_logreg, _ = precision_recall_curve(y_test, y_logreg_pred_prob)
p_knn, r_knn, _ = precision_recall_curve(y_test, y_knn_pred_prob)
p_rf, r_rf, _ = precision_recall_curve(y_test, y_rf_pred_prob)
p_svm, r_svm, _ = precision_recall_curve(y_test, y_svm_pred_prob)
p_ada, r_ada, _ = precision_recall_curve(y_test, y_ada_pred_prob)
p_vc, r_vc, _ = precision_recall_curve(y_test, y_vc_pred_prob)
plt.plot([1,0],[0,1],'k--',color='red')
plt.plot(r_logreg,p_logreg,label='LR')
plt.plot(r_knn,p_knn,label='kNN')
plt.plot(r_rf,p_rf,label='RF')
plt.plot(r_svm,p_svm,label='SVM')
plt.plot(r_ada,p_ada,label='ADA')
plt.plot(r_vc,p_vc,label='VC')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curves - All Models')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=[10,6])
plt.subplot(1,2,1)

plt.plot([0,1],[0,1],'k--',color='blue')
plt.plot(fpr_knn,tpr_knn,label='ROC-AUC: %0.2f'%roc_auc_score(y_test, y_knn_pred_prob))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('kNN ROC Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot([1,0],[0,1],'k--',color='red')
plt.plot(r_knn,p_knn,label='PR AUC: %0.2f'%(auc(r_knn,p_knn)))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('kNN PR Curve')
plt.legend()
plt.tight_layout()

plt.show()
