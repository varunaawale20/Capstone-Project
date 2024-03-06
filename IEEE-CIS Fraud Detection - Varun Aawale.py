#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the necessary ML Libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,
plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA


# In[3]:


#Uploading & Merging the Dataset


# In[ ]:


train_transaction = pd.read_csv(r'C:\Users\varun\Desktop\Project\ieee-fraud-detection\train_transaction.csv', encoding='utf-8', dtype={'column_name': 'float32'})
train_identity = pd.read_csv(r'C:\Users\varun\Desktop\Project\ieee-fraud-detection\train_identity.csv')


# In[ ]:


train_transaction.info()


# In[ ]:


train_identity.info()


# In[ ]:


Merged_data = pd.merge(train_transaction, train_identity, how = 'left', on = ['TransactionID'])


# In[ ]:


Full_data = Merged_data.copy()


# In[ ]:


Full_data.info()


# In[ ]:


#Removing Columns with missing values more than 80%


# In[ ]:


missing_ratio = ((Full_data.isnull().sum() / len(Full_data)))
missing_above80 = missing_ratio[missing_ratio > 0.80].index
Full_data.drop(missing_above80, axis = 1, inplace = True)
Full_data.head()


# In[ ]:


#Date Conversion


# In[ ]:


import datetime
START_DATE = '2022-01-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
Full_data['NewDate'] = Full_data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
Full_data['NewDate_YMD'] = Full_data['NewDate'].dt.year.astype(str) + '-' + Full_data['NewDate'].dt.month.astype(str) + '-' + Full_data['NewDate'].dt.day.astype(str)
Full_data['Month'] = Full_data['NewDate'].dt.year.astype(str) + '-' + Full_data['NewDate'].dt.month.astype(str)
Full_data['Weekday'] = Full_data['NewDate'].dt.dayofweek
Full_data['Hour'] = Full_data['NewDate'].dt.hour
Full_data['Day'] = Full_data['NewDate'].dt.day


# In[ ]:


#Dropping Irrelevant Columns


# In[ ]:


Full_data.drop(["TransactionDT", "NewDate", "NewDate_YMD", "TransactionID"], axis = 1, inplace = True)


# In[ ]:


#Grouping E-mail Values


# In[ ]:


Full_data.loc[Full_data['P_emaildomain'].isin(['gmail.com', 'gmail']),'P_emaildomain'] = 'Google'
Full_data.loc[Full_data['P_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx', 'yahoo.co.uk', 'yahoo.co.jp', 'yahoo.de', 'yahoo.fr', 'yahoo.es']), 'P_emaildomain'] = 'Yahoo Mail'
Full_data.loc[Full_data['P_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 'hotmail.es','hotmail.co.uk', 'hotmail.de', 'outlook.es', 'live.com', 'live.fr', 'hotmail.fr']), 'P_emaildomain'] = 'Microsoft'
Full_data.loc[Full_data['P_emaildomain'].isin(Full_data['P_emaildomain'].value_counts()[Full_data['P_emaildomain'].value_counts() <= 500 ].index), 'P_emaildomain'] = "Others"


# In[ ]:


#Fill Null Values with most common values


# In[ ]:


Full_data = Full_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
Full_data.isnull().sum()


# In[ ]:


#Finding the Class Distribution


# In[ ]:


fraud = Full_data.loc[Full_data['isFraud'] == 1]
non_fraud = Full_data.loc[Full_data['isFraud'] == 0]
Full_data['isFraud'].value_counts(normalize=True)


# In[ ]:


#Summary of the Transaction Amount Distribution


# In[ ]:


print(pd.concat([Full_data['TransactionAmt'].quantile([.01, .1, .25, .5, .75, .9, .99]).reset_index(), fraud['TransactionAmt'].quantile([.01, .1, .25, .5, .75, .9, .99]).reset_index(), non_fraud['TransactionAmt'].quantile([.01, .1, .25, .5, .75, .9, .99]).reset_index()], axis=1, keys=['Total','Fraud', "No Fraud"]))


# In[ ]:


print('Fraud TransactionAmt mean : '+str(fraud['TransactionAmt'].mean()))
print('Non - Fraud TransactionAmt mean: '+str(non_fraud['TransactionAmt'].mean()))


# In[ ]:


plt.figure(figsize=(20,10))
sns.distplot(Full_data["TransactionAmt"].apply(np.log))
plt.title('Train - Test TransactionAmt distribution')
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(fraud["TransactionAmt"].apply(np.log), label = 'Fraud | isFraud = 1')
sns.distplot(non_fraud["TransactionAmt"].apply(np.log), label = 'non-Fraud | isFraud = 0')
plt.title('Fraud vs non-Fraud TransactionAmt distribution')
plt.legend()
plt.show()


# In[ ]:


#Defining Plotting Function


# In[ ]:


def ploting_cnt_amt(df, col, lim=2000):
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    total = len(df)
    
    plt.figure(figsize=(16,14))
    plt.suptitle(f'{col} Distributions ', fontsize=24)
    
    plt.subplot(211)
    g = sns.countplot( x=col, data=df, order=list(tmp[col].values))
    gt = g.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values), color='black', legend=False, )
    gt.set_ylim(0,tmp['Fraud'].max()*1.1)
    gt.set_ylabel("%Fraud Transactions", fontsize=16)
    g.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)
    g.set_xlabel(f"{col} Category Names", fontsize=16)
    g.set_ylabel("Count", fontsize=17)
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    sizes = []
    
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center",fontsize=12)
    g.set_ylim(0,max(sizes)*1.15)
    plt.show()


# In[ ]:


#Performing EDA on all Features


# In[ ]:


ploting_cnt_amt(Full_data, 'P_emaildomain')
Full_data['P_emaildomain'].value_counts()


# In[ ]:


ploting_cnt_amt(Full_data, 'ProductCD')
Full_data['ProductCD'].value_counts()


# In[ ]:


ploting_cnt_amt(Full_data, 'DeviceType')
Full_data['DeviceType'].value_counts()


# In[ ]:


fig,ax = plt.subplots(4, 1, figsize=(16,15))
Full_data.groupby('Weekday')['isFraud'].mean().to_frame().plot.bar(ax=ax[0])
Full_data.groupby('Hour')['isFraud'].mean().to_frame().plot.bar(ax=ax[1])
Full_data.groupby('Day') ['isFraud'].mean().to_frame() .plot.bar(ax=ax[2])
Full_data.groupby('Month')['isFraud'].mean().to_frame().plot.bar (ax=ax[3])


# In[ ]:


#Split Train and Test set


# In[ ]:


# Create X and y arrays
y = Full_data['isFraud'].copy()
X = Full_data.drop('isFraud', axis=1)


# In[ ]:


# Create a train-test split with test size of 0.2
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


y_train = pd.DataFrame(y_train)
y_train.columns = ["isFraud"]
y_train
X_train = pd.DataFrame(X_train, columns = X.columns, index = X_train.index)


# In[ ]:


y_test = pd.DataFrame(y_test)
y_test.columns = ["isFraud"]
y_test
X_test = pd.DataFrame(X_test, columns = X.columns, index = X_test.index )


# In[ ]:


#Split Train and Test Dataset in Categorical and Continuous


# In[ ]:


train_data_cont = X_train.select_dtypes(include = ['int64', 'float64'])
train_data_cat = X_train.select_dtypes(include = 'object')


# In[ ]:


test_data_cont = X_test.select_dtypes(include = ['int64', 'float64'])
test_data_cat = X_test.select_dtypes(include = 'object')


# In[ ]:


#Applying label encoding to categorical features


# In[ ]:


from sklearn.preprocessing import LabelEncoder
for col in train_data_cat.columns:
    if train_data_cat[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(train_data_cat[col].astype(str).values))
        train_data_cat[col] = le.transform(list(train_data_cat[col].astype(str).values))


# In[ ]:


from sklearn.preprocessing import LabelEncoder
for col in test_data_cat.columns:
    if test_data_cat[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(test_data_cat[col].astype(str).values))
        test_data_cat[col] = le.transform(list(test_data_cat[col].astype(str).values))


# In[ ]:


#Standardize the features except the target variable


# In[ ]:


from sklearn.preprocessing import StandardScaler
stndS = StandardScaler()
X_train_std = stndS.fit_transform(train_data_cont)
X_test_std = stndS.transform(test_data_cont)


# In[ ]:


X_train_prepared = pd.DataFrame(X_train_std, columns = train_data_cont.columns, index = train_data_cont.index)
X_test_prepared = pd.DataFrame(X_test_std, columns = test_data_cont.columns, index = test_data_cont.index)


# In[ ]:


#Merging the Categorical and Numerical Features


# In[ ]:


X_train_df = pd.concat([X_train_prepared, train_data_cat], axis = 1)
X_test_df = pd.concat([X_test_prepared, test_data_cat], axis = 1)


# In[ ]:


X_train_df.reset_index(drop=True, inplace=True)
X_train_df


# In[ ]:


X_test_df.reset_index(drop=True, inplace=True)
X_test_df


# In[ ]:


#PCA on Train data


# In[ ]:


X_train_df.reset_index(drop=True, inplace=True)
mylist = [c for c in X_train_df.columns if c[0] == "V"]
vcol=X_train_df[mylist]
pca = PCA(n_components = 3)
pca.fit(vcol)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
vcol1=pca.transform(vcol)
X_train_df.drop(vcol,axis=1,inplace=True)
vcol1 = pd.DataFrame(vcol1)
vcol1.columns = ["V_1", "V_2", "V_3"]


# In[ ]:


X_train_pca = pd.concat([X_train_df, vcol1],axis=1)
X_train_pca


# In[ ]:


mylist_id = [c for c in X_train_pca.columns if c[0:2] == "id"]
idcol=X_train_pca[mylist_id]
pca = PCA(n_components = 3)
pca.fit(idcol)
print(pca.explained_variance_ratio_.sum())
idcol1 = pca.transform(idcol)
X_train_pca.drop(idcol,axis=1,inplace=True)
idcol1 = pd.DataFrame(idcol1)
idcol1.columns = ["id_1", "id_2", "id_3"]
idcol1
X_train_idpca = pd.concat([X_train_pca, idcol1], axis=1)
X_train_idpca


# In[ ]:


mylist_D = [c for c in X_train_idpca.columns if c[0] == "D"]
mylist_D.remove('DeviceType')
mylist_D.remove('DeviceInfo')
Dcol=X_train_pca[mylist_D]
pca = PCA(n_components = 3)
pca.fit(Dcol)
print(pca.explained_variance_ratio_.sum())
Dcol1 = pca.transform(Dcol)
X_train_idpca.drop(Dcol,axis=1,inplace=True)
Dcol1 = pd.DataFrame(Dcol1)
Dcol1.columns = ["D_1", "D_2", "D_3"]
X_train_Dpca = pd.concat([X_train_idpca, Dcol1],axis=1)
X_train_Dpca


# In[ ]:


print(X_train_Dpca.columns.tolist()


# In[ ]:


mylist_M = [c for c in X_train_Dpca.columns if c[0] == "M"]
Mcol=X_train_Dpca[mylist_M]
pca = PCA(n_components = 3)
pca.fit(Mcol)
print(pca.explained_variance_ratio_.sum())
Mcol1 = pca.transform(Mcol)
X_train_Dpca.drop(Mcol,axis=1,inplace=True)
Mcol1 = pd.DataFrame(Mcol1)
Mcol1.columns = ["M_1", "M_2", "M_3"]
X_train_Mpca = pd.concat([X_train_Dpca, Mcol1],axis=1)
X_train_Mpca


# In[ ]:


mylist_C = [c for c in X_train_Mpca.columns if c[0] == "C"]
Ccol=X_train_Mpca[mylist_C]
pca = PCA(n_components = 3)
pca.fit(Ccol)
print(pca.explained_variance_ratio_.sum())
Ccol1 = pca.transform(Ccol)
X_train_Mpca.drop(Ccol,axis=1,inplace=True)
Ccol1 = pd.DataFrame(Ccol1)
Ccol1.columns = ["C_1", "C_2", "C_3"]
X_train_Cpca = pd.concat([X_train_Mpca, Ccol1],axis=1)
X_train_Cpca


# In[ ]:


y_train.reset_index(drop=True, inplace=True)


# In[ ]:


Full_data_train = pd.concat([ y_train, X_train_Cpca], axis = 1)


# In[ ]:


#Plotting Correlation Heatmap


# In[ ]:


plt.figure(figsize=(50, 12))
mask = np.triu(np.ones_like(Full_data_train.corr(), dtype=np.bool))
heatmap = sns.heatmap(Full_data_train.corr(), mask=mask, vmin=-1, vmax=1, annot=True,
cmap='BrBG')
sns.set(font_scale=1.6)
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':60}, pad=16);
plt.savefig('Triangleheatmap.png', dpi=300, bbox_inches='tight')


# In[ ]:


plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(Full_data_train.corr()[['isFraud']].sort_values(by='isFraud', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG') 
heatmap.set_title('Features Correlating with Fraud Values', fontdict={'fontsize':18}, pad=16);
plt.savefig('Features.png', dpi=300, bbox_inches='tight')


# In[ ]:


#Apply SMOT on train dataset


# In[ ]:


sm = SMOTE(random_state = 25, sampling_strategy = 0.5) 

# again we are eqalizing both the classes
# fit the sampling

X_train_Mpca, y_train = sm.fit_resample(X_train_Mpca, y_train)


# In[ ]:


#PCA on test set


# In[ ]:


X_test_df1 = X_test_df.copy()
X_test_df1


# In[ ]:


X_test_df1.reset_index(drop=True, inplace=True)
mylist_V = [c for c in X_test_df1.columns if c[0] == "V"]
vcol_test=X_test_df1[mylist_V]
pca = PCA(n_components = 3)
pca.fit(vcol_test)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
vcol_test1 = pca.transform(vcol_test)
X_test_df1.drop(vcol_test,axis=1,inplace=True)
vcol_test1 = pd.DataFrame(vcol_test1)
vcol_test1.columns = ["V_1", "V_2", "V_3"]
X_test_pca = pd.concat([X_test_df1,vcol_test1],axis=1)
X_test_pca


# In[ ]:


mylist_testid = [c for c in X_test_pca.columns if c[0:2] == "id"]
idcol_test = X_test_pca[mylist_testid]
mylist_testid
pca = PCA(n_components = 3)
pca.fit(idcol_test)
print(pca.explained_variance_ratio_.sum())
idcol_test1 = pca.transform(idcol_test)
X_test_pca.drop(idcol_test,axis=1,inplace=True)
idcol_test1 = pd.DataFrame(idcol_test1)
idcol_test1.columns = ["id_1", "id_2", "id_3"]
idcol_test1
X_test_idpca = pd.concat([X_test_pca, idcol_test1],axis=1)
X_test_idpca


# In[ ]:


mylist_testD = [c for c in X_test_idpca.columns if c[0] == "D"]
mylist_testD.remove('DeviceType')
mylist_testD.remove('DeviceInfo')
Dcol_test = X_test_idpca[mylist_testD]
pca = PCA(n_components = 3)
pca.fit(Dcol_test)
print(pca.explained_variance_ratio_.sum())
Dcol_test1 = pca.transform(Dcol_test)
X_test_idpca.drop(Dcol_test,axis=1,inplace=True)
Dcol_test1 = pd.DataFrame(Dcol_test1)
Dcol_test1.columns = ["D_1", "D_2", "D_3"]
X_test_Dpca = pd.concat([X_test_idpca, Dcol_test1],axis=1)
X_test_Dpca


# In[ ]:


mylist_testM = [c for c in X_test_Dpca.columns if c[0] == "M"]
Mcol_test=X_test_Dpca[mylist_testM]
pca = PCA(n_components = 3)
pca.fit(Mcol_test)
print(pca.explained_variance_ratio_.sum())
Mcol_test1 = pca.transform(Mcol_test)
X_test_Dpca.drop(Mcol_test,axis=1,inplace=True)
Mcol_test1 = pd.DataFrame(Mcol_test1)
Mcol_test1.columns = ["M_1", "M_2", "M_3"]
X_test_Mpca = pd.concat([X_test_Dpca, Mcol_test1],axis=1)
X_test_Mpca


# In[ ]:


#Logistic Regression Model


# In[ ]:


from sklearn.linear_model import LogisticRegression
model_reg = LogisticRegression()
model_reg.fit(X_train_Mpca,y_train)


# In[ ]:


#LR_Training the data


# In[ ]:


from sklearn import metrics
from sklearn.metrics import roc_auc_score
y_train_pred = model_reg.predict(X_train_Mpca)
print('Accuracy of logistic regression classifier on train set:{:.2f}'.format(model_reg.score(X_train_Mpca, y_train)))
print(metrics.classification_report(y_train, y_train_pred))
print(metrics.confusion_matrix(y_train, y_train_pred))
print(roc_auc_score(y_train, y_train_pred))
plot_confusion_matrix(model_reg, X_train_Mpca,y_train)


# In[ ]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred)
plt.figure(figsize=(6,6))
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.title("ROC curve of Logit")
plt.xlabel("False Positive Rate (1-specificity)")
plt.ylabel("True Positive Rate (recall)")
plt.show()


# In[ ]:


AUC = auc(fpr,tpr)


# In[ ]:


#Hyperparameter Tuning on Logistic Regression Model


# In[ ]:


from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


# Create regularization penalty space
penalty = ['l2','l1','elasticnet']


# In[ ]:


# Create regularization hyperparameter distribution using uniform distribution
C = uniform(loc=0, scale=4)


# In[ ]:


Solver = ['newton-cg', 'lbfgs', 'liblinear']


# In[ ]:


# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty, solver=Solver)


# In[ ]:


clf = RandomizedSearchCV(model_reg, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)
best_model = clf.fit(X_train_Mpca, y_train)


# In[ ]:


# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('Best solver:', best_model.best_estimator_.get_params()['solver'])


# In[ ]:


#Predict on Test set


# In[ ]:


y_test_pred = best_model.predict(X_test_Mpca)
print('Accuracy of logistic regression classifier on test set:{:.2f}'.format(model_reg.score(X_test_Mpca, y_test)))


# In[ ]:


print(metrics.classification_report(y_test, y_test_pred))
print(metrics.confusion_matrix(y_test, y_test_pred))
print(roc_auc_score(y_test, y_test_pred))
plot_confusion_matrix(best_model, X_test_Mpca, y_test)


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
plt.figure(figsize=(6,6))
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.title("ROC curve of Logit")
plt.xlabel("False Positive Rate (1-specificity)")
plt.ylabel("True Positive Rate (recall)")
plt.show()


# In[ ]:


#XGBoost Model


# In[ ]:


import xgboost as xgb
model=xgb.XGBClassifier(random_state=145,learning_rate=0.01)
model.fit(X_train_Mpca, y_train)


# In[ ]:


#Computing Train Statistics


# In[ ]:


predictions_xg = model.predict(X_train_Mpca)


# In[ ]:


print(accuracy_score(predictions_xg, y_train))
print('Accuracy score of the XGboost model is {}'.format(accuracy_score(y_train,
predictions_xg)))
print(metrics.classification_report(y_train, predictions_xg))
print(metrics.confusion_matrix(y_train, predictions_xg))
plot_confusion_matrix(model, X_train_Mpca, y_train)


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train, predictions_xg)
plt.figure(figsize=(6,6))
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.title("ROC curve of XGboost")
plt.xlabel("False Positive Rate (1-specificity)")
plt.ylabel("True Positive Rate (recall)")
plt.show()


# In[ ]:


#XGBoost - Hyperparameter tuning


# In[ ]:


hyperparameters_1 = {"subsample":[0.5, 0.75, 1], "colsample_bytree":[0.5, 0.75, 1], "max_depth":[2, 6, 12], "min_child_weight":[1,5,15], "learning_rate":[0.1, 0.3, 0.05], "n_estimators":[100]}


# In[ ]:


clf_1 = RandomizedSearchCV(model, hyperparameters_1, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)
best_model_1 = clf_1.fit(X_train_Mpca, y_train)


# In[ ]:


# View best hyperparameters


# In[ ]:


print('Best subsample:', best_model_1.best_estimator_.get_params()['subsample'])
print('Best colsample_bytree:', best_model_1.best_estimator_.get_params()['colsample_bytree'])
print('Best max_depth:', best_model_1.best_estimator_.get_params()['max_depth'])
print('Best min_child_weight:', best_model_1.best_estimator_.get_params()['min_child_weight'])
print('Best learning_rate:', best_model_1.best_estimator_.get_params()['learning_rate'])
print('Best n_estimators:', best_model_1.best_estimator_.get_params()['n_estimators'])


# In[ ]:


#Predict on Test set


# In[ ]:


predictions_test_xg = best_model_1.predict(X_test_Mpca)
print('Accuracy of XGBoost Classifier on test set:{:.2f}'.format(best_model_1.score(X_test_Mpca, y_test)))


# In[ ]:


print(accuracy_score(predictions_test_xg, y_test))
print('Accuracy score of the XGboost model is {}'.format(accuracy_score(y_test, predictions_test_xg)))
print(metrics.classification_report(y_test, predictions_test_xg))
print(metrics.confusion_matrix(y_test, predictions_test_xg))
plot_confusion_matrix(best_model_1, X_test_Mpca, y_test)


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, predictions_test_xg)
plt.figure(figsize=(6,6))
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.title("ROC curve of XGboost test")
plt.xlabel("False Positive Rate (1-specificity)")
plt.ylabel("True Positive Rate (recall)")
plt.show()


# In[ ]:


#Decision Tree Model


# In[ ]:


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
DT = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
DT.fit(X_train_Mpca, y_train)
DT_pred_train = DT.predict(X_train_Mpca)


# In[ ]:


print(accuracy_score(DT_pred_train, y_train))
print('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(y_train, DT_pred_train)))
print(metrics.classification_report(y_train, DT_pred_train))
print(metrics.confusion_matrix(y_train, DT_pred_train))


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train, DT_pred_train)
plt.figure(figsize=(6,6))
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.title("ROC curve of Decision Tree train set")
plt.xlabel("False Positive Rate (1-specificity)")
plt.ylabel("True Positive Rate (recall)")
plt.show()


# In[ ]:


#Decision Tree - HyperParameter Tuning


# In[ ]:


hyperparameters_2 = {"min_samples_split":[2, 4, 6], "min_samples_leaf":[1, 10, 50], "max_depth":[10,50,70,100], "max_features":['auto', 'sqrt', 'log2'], "criterion":['gini', 'entropy'], "splitter":['random', 'best'], "min_weight_fraction_leaf":[0,0.5]}


# In[ ]:


clf_2 = RandomizedSearchCV(DT, hyperparameters_2, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)
best_model_2 = clf_2.fit(X_train_Mpca, y_train)


# In[ ]:


# View best hyperparameters


# In[ ]:


print('Best min_samples_split:', best_model_2.best_estimator_.get_params()['min_samples_split'])
print('Best min_samples_leaf:', best_model_2.best_estimator_.get_params()['min_samples_leaf'])
print('Best max_depth:', best_model_2.best_estimator_.get_params()['max_depth'])
print('Best max_features:', best_model_2.best_estimator_.get_params()['max_features'])
print('Best criterion:', best_model_2.best_estimator_.get_params()['criterion'])
print('Best splitter:', best_model_2.best_estimator_.get_params()['splitter'])
print('Best min_weight_fraction_leaf:', best_model_2.best_estimator_.get_params()['min_weight_fraction_leaf'])


# In[ ]:


#Decision Tree on Test set


# In[ ]:


predictions_test_DT = best_model_2.predict(X_test_Mpca)


# In[ ]:


print(accuracy_score(DT_pred_train, y_train))
print('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(y_test,
predictions_test_DT)))
print(metrics.classification_report(y_test, predictions_test_DT))
print(metrics.confusion_matrix(y_test, predictions_test_DT))


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, predictions_test_DT)
plt.figure(figsize=(6,6))
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.title("ROC curve of Decision Tree test")
plt.xlabel("False Positive Rate (1-specificity)")
plt.ylabel("True Positive Rate (recall)")
plt.show()


# In[ ]:


#Random Forest Model


# In[ ]:


rf = RandomForestClassifier(max_depth = 4)
rf.fit(X_train_Mpca, y_train)
rf_pred_train = rf.predict(X_train_Mpca)


# In[ ]:


print(accuracy_score(rf_pred_train, y_train))
print('Accuracy score of the Random Forest on train set is{}'.format(accuracy_score(y_train, rf_pred_train)))
print(metrics.classification_report(y_train, rf_pred_train))
print(metrics.confusion_matrix(y_train, rf_pred_train))


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train, rf_pred_train)
plt.figure(figsize=(6,6))
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.title("ROC curve of Random Forest train")
plt.xlabel("False Positive Rate (1-specificity)")
plt.ylabel("True Positive Rate (recall)")
plt.show()


# In[ ]:


#Random Forest - Hyperparameter Tuning


# In[ ]:


Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# In[ ]:


random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}


# In[ ]:


clf_3 = RandomizedSearchCV(rf, random_grid, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)
best_model_3 = clf_3.fit(X_train_Mpca, y_train)


# In[ ]:


# View best hyperparameters
print('Best n_estimators:', best_model_3.best_estimator_.get_params()['n_estimators'])
print('Best max_features:', best_model_3.best_estimator_.get_params()['max_features'])
print('Best max_depth:', best_model_3.best_estimator_.get_params()['max_depth'])
print('Best min_samples_split:', best_model_3.best_estimator_.get_params()['min_samples_split'])
print('Best min_samples_leaf:', best_model_3.best_estimator_.get_params()['min_samples_leaf'])
print('Best bootstrap:', best_model_3.best_estimator_.get_params()['bootstrap'])


# In[ ]:


#Predict on Test set


# In[ ]:


rf_pred_test = best_model_3.predict(X_test_Mpca)
print(accuracy_score(rf_pred_test, y_test))
print('Accuracy score of the Random Forest model is {}'.format(accuracy_score(y_test, rf_pred_test)))
print(metrics.classification_report(y_test, rf_pred_test))
print(metrics.confusion_matrix(y_test, rf_pred_test))


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, rf_pred_test)
plt.figure(figsize=(6,6))
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.title("ROC curve of Random Forest")
plt.xlabel("False Positive Rate (1-specificity)")
plt.ylabel("True Positive Rate (recall)")
plt.show()


# In[ ]:


#SVM Model


# In[ ]:


from sklearn.svm import LinearSVC
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train_Mpca, y_train)
svm_pred = svm.predict(X_train_Mpca)


# In[ ]:


print(accuracy_score(svm_pred, y_train))
print('Accuracy score of the Support Vector Machine model is{}'.format(accuracy_score(y_train, svm_pred)))
print(metrics.classification_report(y_train, svm_pred))
print(metrics.confusion_matrix(y_train, svm_pred))


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train, svm_pred)
plt.figure(figsize=(6,6))
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.title("ROC curve of Random Forest train")
plt.xlabel("False Positive Rate (1-specificity)")
plt.ylabel("True Positive Rate (recall)")
plt.show()


# In[ ]:


#SVM - HyperParameter Tuning


# In[ ]:


from scipy.stats import reciprocal, uniform
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}


# In[ ]:


clf_4 = RandomizedSearchCV(svm, param_grid, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)
best_model_4 = clf_4.fit(X_train_Mpca, y_train)


# In[ ]:


# View best hyperparameters
print('Best C:', best_model_4.best_estimator_.get_params()['C'])
print('Best gamma:', best_model_4.best_estimator_.get_params()['gamma'])


# In[ ]:


#Predict on SVM Model


# In[ ]:


svm_pred_test = svm.predict(X_test_Mpca)


# In[ ]:


print(accuracy_score(svm_pred_test, y_test))
print('Accuracy score of the Support Vector Machine model is
{}'.format(accuracy_score(y_test, svm_pred_test)))
print(metrics.classification_report(y_test, svm_pred_test))
print(metrics.confusion_matrix(y_test, svm_pred_test))


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, svm_pred_test)
plt.figure(figsize=(6,6))
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.title("ROC curve of Random Forest")
plt.xlabel("False Positive Rate (1-specificity)")
plt.ylabel("True Positive Rate (recall)")
plt.show()


# In[ ]:


#ANN Model


# In[ ]:


import keras
from keras.layers import Dense
from keras.models import load_model
from keras.models import Sequential
classifier = Sequential()
# Defining the Input layer and FIRST hidden layer,both are same!
# relu means Rectifier linear unit function
classifier.add(Dense(units=100, input_dim=45, kernel_initializer='uniform', activation='relu'))


# In[ ]:


#Defining the SECOND hidden layer, here we have not defined input because it is
# second layer and it will get input as the output of first hidden layer
classifier.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))


# In[ ]:


# Defining the Output layer
# sigmoid means sigmoid activation function
# for Multiclass classification the activation ='softmax'
# And output_dim will be equal to the number of factor levels
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))


# In[ ]:


# Optimizer== the algorithm of SGG to keep updating weights
# loss== the loss function to measure the accuracy
# metrics== the way we will compare the accuracy after each step of SGD
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


# fitting the Neural Network on the training data
ANN_Model=classifier.fit(X_train_Mpca,y_train, batch_size=100 , epochs=20, verbose=1)


# In[ ]:


# Predictions on testing data
Predictions_train_ANN=classifier.predict(X_train_Mpca)
Predictions_train_ANN = pd.DataFrame(Predictions_train_ANN, columns=["isFraud"])


# In[ ]:


# Defining the probability threshold
def probThreshold(inpProb):
    if inpProb > 0.6:
        return(1)
    else:
        return(0)


# In[ ]:


Predictions_train_ANN['isFraud'] =Predictions_train_ANN['isFraud'].apply(probThreshold)
Predictions_train_ANN
type(Predictions_train_ANN)


# In[ ]:


print(accuracy_score(Predictions_train_ANN, y_train))
print('Accuracy score of the ANN model is {}'.format(accuracy_score(y_train, Predictions_train_ANN)))


# In[ ]:


print(metrics.classification_report(y_train, Predictions_train_ANN))
print(metrics.confusion_matrix(y_train, Predictions_train_ANN))


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train, Predictions_train_ANN)
plt.figure(figsize=(6,6))
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.title("ROC curve of ANN")
plt.xlabel("False Positive Rate (1-specificity)")
plt.ylabel("True Positive Rate (recall)")
plt.show()


# In[ ]:


#Prediction on Test set


# In[ ]:


# Predictions on testing data
Predictions_test_ANN=classifier.predict(X_test_Mpca)
Predictions_test_ANN = pd.DataFrame(Predictions_test_ANN, columns=["isFraud"])


# In[ ]:


# Defining the probability threshold
def probThreshold(inpProb):
    if inpProb > 0.6:
        return(1)
    else:
        return(0)


# In[ ]:


Predictions_test_ANN['isFraud'] =Predictions_test_ANN['isFraud'].apply(probThreshold)
Predictions_test_ANN
type(Predictions_test_ANN)


# In[ ]:


print(accuracy_score(Predictions_test_ANN, y_test))
print('Accuracy score of the ANN model is {}'.format(accuracy_score(y_test, Predictions_test_ANN)))
print(metrics.classification_report(y_test, Predictions_test_ANN))
print(metrics.confusion_matrix(y_test, Predictions_test_ANN))


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, Predictions_test_ANN)
plt.figure(figsize=(6,6))
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.title("ROC curve of ANN")
plt.xlabel("False Positive Rate (1-specificity)")
plt.ylabel("True Positive Rate (recall)")
plt.show()


# In[ ]:


X_train_Mpca_1 = np.zeros(X_train_Mpca.shape,dtype='uint8')


# In[ ]:


y_train.values_1 = np.zeros(y_train.values.shape,dtype='uint8')

