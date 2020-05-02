# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:56:49 2020

@author: raddoda
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report 
plt.rc("font", size=14)

# Libraries for data modelling
from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Common sklearn Model Helpers
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
# from sklearn.datasets import make_classification

# sklearn modules for performance metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import average_precision_score


# Importing the dataset
dataset = pd.read_csv('blood_donation.csv')
dataset['Donation_status'].value_counts()
plt.figure(figsize=(20,7))
corrMatrix = dataset.corr()
sn.heatmap(corrMatrix, annot=True)

X = dataset.drop(['Donation_status'], axis=1)
y = dataset['Donation_status']


#get col names
names=X.columns

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scale= sc.fit_transform(X)
X_new=pd.DataFrame(X_scale,columns=names)



# Applying PCA
# from sklearn.decomposition import PCA
#pca = PCA(n_components = 6)
#X_pca= pca.fit_transform(X_scale)

#explained_variance = pca.explained_variance_ratio_

#plt.figure(figsize=(15,5))
#corrMatrix = data.corr()
#sn.heatmap(corrMatrix, annot=True)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)


# Fitting Logistic Regression to the Training set
(np.random.seed(1234))
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test,y_pred)

#----------------------------------------------
#            SVM
#-----------------------------------------------
(np.random.seed(1234))
from sklearn.svm import SVC
from sklearn import svm
model=svm.SVC()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))


#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test,y_pred)

#----------------------------------------------
#            
#-----------------------------------------------
#----------------------------------------------
#            RandomForest
#-----------------------------------------------
(np.random.seed(1234))
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
roc_auc_score(y_test,y_pred)

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test,y_pred)
#----------------------------------------------
#            XGBOOST
#-----------------------------------------------
(np.random.seed(1234))
from xgboost import XGBClassifier
xg=XGBClassifier()
xg.fit(X_train,y_train)
# Predicting the Test set results
y_pred = xg.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test,y_pred)


roc_auc_score(y_test,y_pred)

#--------------------------------------------------
# classifiers
#----------------------------------------------------

# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1234), 
               svm.SVC(),  
               RandomForestClassifier(random_state=1234),
               XGBClassifier()]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
#Plot the figure
fig = plt.figure(figsize=(6,5))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.2f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("FPR", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("TPR", fontsize=15)

plt.title('Baseline Model: ROC Curve Analysis', fontweight='bold', fontsize=13)
plt.legend(prop={'size':11}, loc='lower right')

plt.show()

#----------------------------------------------------
#                    SMOTE METHOD
#---------------------------------------------------

from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_smote, y_smote = sm.fit_sample(X, y)
#from collections import Counter
#rint(sorted(Counter(y_russ).items()))
#X_rus.shape,y_russ.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_s= sc.fit_transform(X_smote)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_s, y_smote, test_size = 0.2, random_state = 0)

# Fitting Logistic Regression to the Training set
(np.random.seed(1234))
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test,y_pred)

#----------------------------------------------
#            SVM
#-----------------------------------------------
(np.random.seed(1234))
from sklearn.svm import SVC
from sklearn import svm
model=svm.SVC()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))


#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test,y_pred)

#----------------------------------------------
#            
#-----------------------------------------------
#----------------------------------------------
#            RandomForest
#-----------------------------------------------
(np.random.seed(1234))
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
roc_auc_score(y_test,y_pred)

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test,y_pred)
#----------------------------------------------
#            XGBOOST
#-----------------------------------------------
(np.random.seed(1234))
from xgboost import XGBClassifier
xg=XGBClassifier()
xg.fit(X_train,y_train)
# Predicting the Test set results
y_pred = xg.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test,y_pred)


roc_auc_score(y_test,y_pred)

#--------------------------------------------------
# classifiers
#----------------------------------------------------

# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1234), 
               svm.SVC(),  
               RandomForestClassifier(random_state=1234),
               XGBClassifier()]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
#Plot the figure
fig = plt.figure(figsize=(6,5))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.2f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("FPR", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("TPR", fontsize=15)

plt.title('SMOTE: ROC Curve Analysis', fontweight='bold', fontsize=13)
plt.legend(prop={'size':11}, loc='lower right')

plt.show()


#----------------------------------------------------
#                    ADASYN METHOD
#---------------------------------------------------
(np.random.seed(1234))

from imblearn.over_sampling import ADASYN 
sa = ADASYN()
X_ad, y_ad = sa.fit_sample(X, y)
#from collections import Counter
#rint(sorted(Counter(y_russ).items()))
#X_rus.shape,y_russ.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler
ad = StandardScaler()
X_ads= ad.fit_transform(X_ad)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_ads, y_ad, test_size = 0.2, random_state = 0)


# Fitting Logistic Regression to the Training set
(np.random.seed(1234))
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test,y_pred)

#----------------------------------------------
#            SVM
#-----------------------------------------------
(np.random.seed(1234))
from sklearn.svm import SVC
from sklearn import svm
model=svm.SVC()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))


#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test,y_pred)

#----------------------------------------------
#            
#-----------------------------------------------
#----------------------------------------------
#            RandomForest
#-----------------------------------------------
(np.random.seed(1234))
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
roc_auc_score(y_test,y_pred)

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test,y_pred)
#----------------------------------------------
#            XGBOOST
#-----------------------------------------------
(np.random.seed(1234))
from xgboost import XGBClassifier
xg=XGBClassifier()
xg.fit(X_train,y_train)
# Predicting the Test set results
y_pred = xg.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test,y_pred)


roc_auc_score(y_test,y_pred)

#--------------------------------------------------
# classifiers
#----------------------------------------------------

# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1234), 
               svm.SVC(),  
               RandomForestClassifier(random_state=1234),
               XGBClassifier()]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
#Plot the figure
fig = plt.figure(figsize=(6,5))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.2f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("FPR", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("TPR", fontsize=15)

plt.title('ADASYN: ROC Curve Analysis', fontweight='bold', fontsize=13)
plt.legend(prop={'size':11}, loc='lower right')

plt.show()


