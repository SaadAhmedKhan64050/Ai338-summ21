
***Logistic Regression Implementation***

from numpy import mean
from numpy import std
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#Read CSV Files
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

#Get Labels and remove them from Train data
YTrain = trainData.Survived;
trainData.drop('Survived',inplace=True,axis=1)

#Convert Data to Features by taking 3 sets of features of each test and train 
trainData.drop('Name',inplace=True,axis=1)
trainData.drop('Ticket',inplace=True,axis=1)
trainData.drop('Cabin',inplace=True,axis=1)
testData.drop('Name',inplace=True,axis=1)
testData.drop('Ticket',inplace=True,axis=1)
testData.drop('Cabin',inplace=True,axis=1)
#Convert String based columns to integer classes
trainData["Sex"] = trainData["Sex"].replace(['female','male'],[0,1])
testData["Sex"] = testData["Sex"].replace(['female','male'],[0,1])
trainData["Embarked"] = trainData["Embarked"].replace(['S','Q','C'],[0,1,2])
testData["Embarked"] = testData["Embarked"].replace(['S','Q','C'],[0,1,2])

#Remove Nan Values from Train
trainData.fillna(value=94,inplace=True)
#Dummy values in Test for all NaN
testData.fillna(value=trainData['Age'].mean(),inplace=True)
testData.fillna(value=trainData['Fare'].mean(),inplace=True)

#Test
print(YTrain.shape)
print(trainData)
print(testData.shape)


##Apply Linear Classifier(Logistic Regression)
model = LogisticRegression()
#Train The model
model.fit(trainData,YTrain)

#Here we applied the K-Fold crossValidation for evaluating the accuracy of model 
cv = KFold(n_splits=10, random_state=1, shuffle=True)
##Model Evaluation
scores = cross_val_score(model_Lr, trainData, YTrain, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": predict
    })
submission.to_csv('LR_Submission.csv', index=False)
from google.colab import files
files.download('LR_Submission.csv')
