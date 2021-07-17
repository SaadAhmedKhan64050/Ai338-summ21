import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd





df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Set our train data according to frames
train = pd.DataFrame(df)
df.head()
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()

inputs = df.drop('Survived',axis='columns')
target = df.Survived
#inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})

dummies = pd.get_dummies(inputs.Sex)
dummies.head(10)
inputs = pd.concat([inputs,dummies],axis='columns')
inputs.head(10)

inputs.drop(['Sex','male'],axis='columns',inplace=True)
inputs.head(10)

inputs.columns[inputs.isna().any()]
inputs.Age[:]
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)
from sklearn.naive_bayes import MultinomialNB

X_test1 = test.drop('PassengerId',axis='columns')
X_test2 = test.PassengerId
model = MultinomialNB()

model.fit(X_train,y_train)
print(f'Score: {model.score(X_test,y_test)}')

X_test[0:]
#model.predict(X_test[0:])
#model.predict_proba(X_test[:])
predict_class = model.predict(X_test)

submission=pd.DataFrame({"PassengerId": list(range(1,len(predict_class)+1)), "Survival": predict_class})
submission.to_csv('submission.csv', index=False,header=True)
from google.colab import files
files.download('submission.csv')