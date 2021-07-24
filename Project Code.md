***Logistic Regression Implementation***<br/>
----------------------------------------------
-----------------------------***Code By Saad Ahmed Khan***-----------------------------------

#Importing the necessary libararies </br>
from numpy import mean <br/>
from numpy import std </br>
import pandas as pd</br>
from sklearn.linear_model import LogisticRegression </br>
import numpy as np </br>
from sklearn.model_selection import KFold </br>
from sklearn.model_selection import cross_val_score </br>

#Read CSV Files </br>
trainData = pd.read_csv('train.csv')</br>
testData = pd.read_csv('test.csv')</br>

#Get Labels and remove them from Train data</br>
YTrain = trainData.Survived;</br>
trainData.drop('Survived',inplace=True,axis=1)</br>

#Convert Data to Features by taking 3 sets of features of each test and train </br>
trainData.drop('Name',inplace=True,axis=1)</br>
trainData.drop('Ticket',inplace=True,axis=1)</br>
trainData.drop('Cabin',inplace=True,axis=1)</br>
testData.drop('Name',inplace=True,axis=1)</br>
testData.drop('Ticket',inplace=True,axis=1)</br>
testData.drop('Cabin',inplace=True,axis=1)</br>
#Convert String based columns to integer classes</br>
trainData["Sex"] = trainData["Sex"].replace(['female','male'],[0,1])</br>
testData["Sex"] = testData["Sex"].replace(['female','male'],[0,1])</br>
trainData["Embarked"] = trainData["Embarked"].replace(['S','Q','C'],[0,1,2])</br>
testData["Embarked"] = testData["Embarked"].replace(['S','Q','C'],[0,1,2])</br>

#Remove Nan Values from Train </br>
trainData.fillna(value=94,inplace=True) </br>
#Dummy values in Test for all NaN </br>
testData.fillna(value=trainData['Age'].mean(),inplace=True) </br>
testData.fillna(value=trainData['Fare'].mean(),inplace=True) </br>

#Test </br>
print(YTrain.shape) </br>
print(trainData)</br>
print(testData.shape) </br>


##Apply Linear Classifier(Logistic Regression) </br>
model = LogisticRegression()</br>
#Train The model </br>
model.fit(trainData,YTrain) </br>

#Here we applied the K-Fold crossValidation for evaluating the accuracy of model </br>
cv = KFold(n_splits=10, random_state=1, shuffle=True)</br>
##Model Evaluation</br>
scores = cross_val_score(model_Lr, trainData, YTrain, scoring='accuracy', cv=cv, n_jobs=-1)</br>
#report performance </br>
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores))) </br>

submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": predict
    })</br>
submission.to_csv('LR_Submission.csv', index=False) </br>
from google.colab import files</br>
files.download('LR_Submission.csv')</br>


***Impleament K Nearest Neighbour Classifier(KNN)***
-----------------------------------------------------------
--------------------------------***Code Emaan Ayesha***-----------------------------------------<br/>

#Implementing KNN <br/>
from numpy import mean <br/>
from numpy import std <br/>
import pandas as pd <br/>
from sklearn.neighbors import KNeighborsClassifier <br/>
import numpy as np <br/>
from sklearn.model_selection import KFold <br/>
from sklearn.model_selection import cross_val_score <br/>

#Read CSV Files
trainData = pd.read_csv('train.csv') <br/>
testData = pd.read_csv('test.csv') <br/>

#Get Labels and remove them from Train data <br/>
YTrain = trainData.Survived;<br/>
trainData.drop('Survived',inplace=True,axis=1) <br/>

#Convert Data to Features <br/>
trainData.drop('Name',inplace=True,axis=1) <br/>
trainData.drop('Cabin',inplace=True,axis=1) <br/>
trainData.drop('Ticket',inplace=True,axis=1) <br/>
testData.drop('Name',inplace=True,axis=1)<br/>
testData.drop('Cabin',inplace=True,axis=1)<br/>
testData.drop('Ticket',inplace=True,axis=1)<br/>
#Convert String based columns to integer classes<br/>
trainData["Sex"] = trainData["Sex"].replace(['female','male'],[0,1]) <br/>
testData["Sex"] = testData["Sex"].replace(['female','male'],[0,1])<br/>
trainData["Embarked"] = trainData["Embarked"].replace(['S','Q','C'],[0,1,2])<br/>
testData["Embarked"] = testData["Embarked"].replace(['S','Q','C'],[0,1,2])<br/>

#Remove Nan Values from Train <br/>
trainData.fillna(value=1,inplace=True) <br/>
#Dummy values in Test for all NaN<br/>
testData.fillna(value=trainData['Age'].mean(),inplace=True) <br/>
testData.fillna(value=trainData['Fare'].mean(),inplace=True) <br/>

#Test<br/>
print(YTrain.shape)<br/>
print(trainData)<br/>
print(testData.shape)

##Applying KNN Model <br/>
model_param = {
      'KNeighborsClassifier':{
        'model':KNeighborsClassifier(),
        'param':{
            'n_neighbors': [5,10,15,20,25]
}
      }
}<br/>
#Evaluating the model <br/>
model_Knn = KNeighborsClassifier( n_neighbors= 25)<br/>
#Train the model<br/>
model_Knn.fit(trainData, YTrain)<br/>

#Applying Kfold crossValidation procedure<br/>
cv = KFold(n_splits=36, random_state=1, shuffle=True)<br/>
##Model Evaluation<br/>
scores = cross_val_score(model_Knn, trainData, YTrain, scoring='accuracy', cv=cv, n_jobs=-1)<br/>
#report performance <br/>
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))<br/>
#Predictions<br/>
predictions = model_Knn.predict(testData)<br/>
print(predictions.shape)<br/>
submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": predictions
    })<br/>
submission.to_csv('Knn_Submission.csv', index=False)<br/>
from google.colab import files</br>
files.download('Knn_Submission.csv')</br>

***Implementing SVM***<br/>
----------------------------------------------
-----------------------------***Code By Areeba Hussain 64057***-----------------------------

#Implementing SVM

from numpy import mean </br>
from numpy import std </br>
import pandas as pd </br>
from sklearn import svm </br>
import numpy as np </br>
from sklearn.model_selection import KFold </br>
from sklearn.model_selection import cross_val_score </br>
</br>

#Read CSV Files </br>
trainData = pd.read_csv('train.csv') </br>
testData = pd.read_csv('test.csv') </br>
</br>

#Get Labels and remove them from Train data </br>
YTrain = trainData.Survived; </br>
trainData.drop('Survived',inplace=True,axis=1) </br>
</br>

#Convert Data to Features </br>
trainData.drop('Name',inplace=True,axis=1) </br>
trainData.drop('Cabin',inplace=True,axis=1) </br>
trainData.drop('Ticket',inplace=True,axis=1) </br>
testData.drop('Name',inplace=True,axis=1) </br>
testData.drop('Cabin',inplace=True,axis=1) </br>
testData.drop('Ticket',inplace=True,axis=1) </br>
#Convert String based columns to integer classes </br>
trainData["Sex"] = trainData["Sex"].replace(['female','male'],[0,1]) </br>
testData["Sex"] = testData["Sex"].replace(['female','male'],[0,1]) </br>
trainData["Embarked"] = trainData["Embarked"].replace(['S','Q','C'],[0,1,2]) </br>
testData["Embarked"] = testData["Embarked"].replace(['S','Q','C'],[0,1,2]) </br>
 </br>
 
#Remove Nan Values from Train </br>
trainData.fillna(value=1,inplace=True) </br>
#Dummy values in Test for all NaN </br>
testData.fillna(value=trainData['Age'].mean(),inplace=True) </br>
testData.fillna(value=trainData['Fare'].mean(),inplace=True) </br>
</br>

#Test </br>
print(YTrain.shape) </br>
print(trainData) </br>
print(testData.shape) </br>
</br>

#Applying SVM model </br>
model=svm.SVC(kernel='rbf',C=1,gamma=0.1) </br>
model.fit(trainData,YTrain) </br>
</br>

#Applying Kfold crossValidation procedure </br>
cv = KFold(n_splits=10, random_state=1, shuffle=True) </br>
##Model Evaluation </br>
scores = cross_val_score(model, trainData, YTrain, scoring='accuracy', cv=cv, n_jobs=-1) </br>
# report performance </br>
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores))) </br>
#Predictions </br>
predictions = model.predict(testData) </br>
print(predictions.shape) </br>
submission = pd.DataFrame </br>
({ </br>
        "PassengerId": testData["PassengerId"], </br>
        "Survived": predictions </br>
        }) </br>
submission.to_csv('SVM_Submission.csv', index=False) </br>
from google.colab import files </br>
files.download('SVM_Submission.csv') </br>
</br>
