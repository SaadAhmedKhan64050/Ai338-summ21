# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1baFot7UVojpLvMOVhwpVZT4JSd0s3ab-
"""

#Implementing Linear Classifier(Logistic Regression)
#Code By Saad Ahmed Khan

#Importing the necessary libraries
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

#Predictions
predictions = model.predict(testData)
print(predictions.shape)

submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('LR_Submission.csv', index=False)
##For Downloading the file
from google.colab import files
files.download('LR_Submission.csv')

#Implementing SVM
#Code By Areeba Hussain

from numpy import mean
from numpy import std
import pandas as pd
from sklearn import svm 
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#Read CSV Files
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

#Get Labels and remove them from Train data
YTrain = trainData.Survived;
trainData.drop('Survived',inplace=True,axis=1)

#Convert Data to Features
trainData.drop('Name',inplace=True,axis=1)
trainData.drop('Cabin',inplace=True,axis=1)
trainData.drop('Ticket',inplace=True,axis=1)
testData.drop('Name',inplace=True,axis=1)
testData.drop('Cabin',inplace=True,axis=1)
testData.drop('Ticket',inplace=True,axis=1)
#Convert String based columns to integer classes
trainData["Sex"] = trainData["Sex"].replace(['female','male'],[0,1])
testData["Sex"] = testData["Sex"].replace(['female','male'],[0,1])
trainData["Embarked"] = trainData["Embarked"].replace(['S','Q','C'],[0,1,2])
testData["Embarked"] = testData["Embarked"].replace(['S','Q','C'],[0,1,2])

#Remove Nan Values from Train
trainData.fillna(value=1,inplace=True)
#Dummy values in Test for all NaN
testData.fillna(value=trainData['Age'].mean(),inplace=True)
testData.fillna(value=trainData['Fare'].mean(),inplace=True)

#Test
print(YTrain.shape)
print(trainData)
print(testData.shape)

#Applying SVM model
model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(trainData,YTrain)

#Applying Kfold crossValidation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
##Model Evaluation
scores = cross_val_score(model, trainData, YTrain, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))    ##Calculating the mean and the standard deviation by using numpy library for getting an accuracy 
#Predictions
predictions = model.predict(testData)
print(predictions.shape)
submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('SVM_Submission.csv', index=False)

##For Downloading the file
from google.colab import files
files.download('SVM_Submission.csv')

#Implementing KNN
#Code By Emaan Ayesha

from numpy import mean
from numpy import std
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#Read CSV Files
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

#Get Labels and remove them from Train data
YTrain = trainData.Survived;
trainData.drop('Survived',inplace=True,axis=1)

#Convert Data to Features
trainData.drop('Name',inplace=True,axis=1)
trainData.drop('Cabin',inplace=True,axis=1)
trainData.drop('Ticket',inplace=True,axis=1)
testData.drop('Name',inplace=True,axis=1)
testData.drop('Cabin',inplace=True,axis=1)
testData.drop('Ticket',inplace=True,axis=1)
#Convert String based columns to integer classes
trainData["Sex"] = trainData["Sex"].replace(['female','male'],[0,1])
testData["Sex"] = testData["Sex"].replace(['female','male'],[0,1])
trainData["Embarked"] = trainData["Embarked"].replace(['S','Q','C'],[0,1,2])
testData["Embarked"] = testData["Embarked"].replace(['S','Q','C'],[0,1,2])

#Remove Nan Values from Train
trainData.fillna(value=1,inplace=True)
#Dummy values in Test for all NaN
testData.fillna(value=trainData['Age'].mean(),inplace=True)
testData.fillna(value=trainData['Fare'].mean(),inplace=True)

#Test
print(YTrain.shape)
print(trainData)
print(testData.shape)

##Applying KNN Model
model_param = {
      'KNeighborsClassifier':{
        'model':KNeighborsClassifier(),
        'param':{
            'n_neighbors': [5,10,15,20,25]
}
      }
}
#Evaluating the model
model_Knn = KNeighborsClassifier( n_neighbors= 25)

#Train the model
model_Knn.fit(trainData, YTrain)

#Applying Kfold crossValidation procedure
cv = KFold(n_splits=36, random_state=1, shuffle=True)

##Model Evaluation
scores = cross_val_score(model_Knn, trainData, YTrain, scoring='accuracy', cv=cv, n_jobs=-1)

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#Predictions
predictions = model_Knn.predict(testData)
print(predictions.shape)

submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('Knn_Submission.csv', index=False)

