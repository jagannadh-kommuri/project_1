import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
data=pd.read_csv("/Users/nithish/Downloads/Iris.csv")
print(data.info)
print(data.head())
print(data.describe())
print(data.describe().T)
print(data.dtypes)


##...Checking the missing values ...##
print(data.isna().sum())


##convert the object into the numerical...#
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Species']=le.fit_transform(data['Species'])
print(data.dtypes)
print(data.head())


##...EDA..##
#data.hist()
#plt.show()
#sns.pairplot(data)
#plt.show()
"""
plt.subplot(2,2,1)
plt.scatter(x='SepalLengthCm',y='Species',data=data)
plt.show()
plt.subplot(2,2,2)
plt.scatter(x='SepalWidthCm',y='Species',data=data)
plt.show()
plt.subplot(2,2,3)
plt.scatter(x='PetalLengthCm',y='Species',data=data)
plt.show()
plt.subplot(2,2,4)
plt.scatter(x='PetalWidthCm',y='Species',data=data)
plt.show()



plt.subplot(2,3,1)
sns.distplot(data['SepalLengthCm'])

plt.subplot(2,3,2)
sns.distplot(data['SepalWidthCm'])

plt.subplot(2,3,3)
sns.distplot(data['PetalLengthCm'])

plt.subplot(2,3,4)
sns.distplot(data['PetalWidthCm'])

plt.show()


sns.countplot(data['Species'])
plt.show()
"""


#split the data for and assign to x&y ...##
x=data.iloc[:,:-1]
y=data.iloc[:,-1:]
print(y.head())
print(y)

###....training the testing...###
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=3,test_size=0.33)
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)



###... Select the model....##
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(xtrain,ytrain)
ypred=log_reg.predict(xtest)
print(ypred)


###...Performance Testing....###
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score,classification_report
print("The confusion matrix:\n",confusion_matrix(ytest,ypred))
print("============================================================")
print("The accuracy score:\t",accuracy_score(ytest,ypred))
print("=======================================================")
print("The classification report:\n",classification_report(ytest,ypred))
print("===============================================================")


###.......Random forest........###
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=15)
rf_model.fit(xtrain,ytrain)
ypred_rf=rf_model.predict(xtest)
#print(ypred_rf)


##performance test for the random forest ##
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score
print("The confusion matrix:\n",confusion_matrix(ytest,ypred_rf))
print("============================================================")
print("The accuracy score:\t",accuracy_score(ytest,ypred_rf))
print("=======================================================")
print("The classification report:\n",classification_report(ytest,ypred_rf))
print("====================================================")



