#import dataset
bank_fraud=pd.read_csv("sweedish_fraud.csv")

#libraries 
import pandas as pd 
import sklearn as sk 
import matplotlib.pyplot as plt 
import random

#visualize feature scaled data
bank_fraud[bank_fraud.dtypes[(bank_fraud.dtypes=="float64") | (bank_fraud.dtypes=="int64")].
index.values].hist(figsize=[5,5])
plt.show() 

#count fraud occurrences
bank_fraud.groupby('isFraud').count() #summary statistics (no fraud vs. no fraud) 
bank_fraud['isFraud'].value_counts() #1047638 cases of no fraud, 937 cases of fraud

#split the data into test and train datasets 
sample=np.random.rand(len(bank_fraud))<0.8
train=bank_fraud[sample] 
test=bank_fraud[~sample]

X_train=train[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']]
y_train=train['isFraud']
X_test=test[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']]
y_test=test['isFraud']

#1. KNN Classifier (Feature Scaling)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

#check KNN model performance
from sklearn.metrics import accuracy_score
accuracy_score(y_test,knn.predict(X_test)) 

#2. Feature Standardization
#standardize the train and test dataset 
from sklearn.preprocessing import scale
X_train_scale=(X_train)
X_test_scale=(X_test)

#logistic regression (with standarized data)
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(X_train_scale,y_train)
#check the logistic regression model accuracy
accuracy_score(y_test,log_model.predict(X_test_scale)) #99.9% accuracy

##summary statistics (fraud dataset)
# distibution original amount 
bank_fraud['amount'].hist(normed=True)
plt.show() 

#count bank misclass frauds? (never becuase never make a fraud claim)
bank_fraud.groupby('isFlaggedFraud').count() #summary statistics (no fraud vs. no fraud) 
bank_fraud['isFlaggedFraud'].value_counts()

#visualizations******** 
bank_fraud.head() 
bank_fraud.describe() 
bank_fraud.info() 

#is data fraud?
bank_fraud['isFlaggedFraud'].sum() #no detection
bank_fraud.drop(['nameOrig','nameDest','isFlaggedFraud'],axis=1,inplace=True)
sns.countplot(bank_fraud['type'],hue=bank_fraud['isFraud'])
plt.show() 

#finding which types of transcations are fradulent?
fraud_transfer=bank_fraud[(bank_fraud.isFraud==1) & (bank_fraud.type=="TRANSFER")]
fraud_transfer
len(fraud_transfer) 

fraud_cashout=bank_fraud[(bank_fraud.isFraud==1) & (bank_fraud.type=="CASH_OUT")]
len(fraud_cashout) 

#amount by type of transaction
bank_fraud.groupby(['type']).mean() 
bank_fraud.pivot_table(index=['type'],aggfunc='mean')

#amount by type (histogram)
bank_fraud['amount'].hist(by=bank_fraud['type'])
plt.show() 
plt.title('Transfer Transaction Are High Money')
plt.xlabel('amount')