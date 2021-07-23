 # Project Report
----------------------------------------------------------------------------------------------------------------------------

***Findings Of all Three Models***
-------------------------------------------------------------------------------------------------------------------------

***Linear Classifier(Logistic Regression)***: <br/>
Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Some of the examples of classification problems are Email spam 
or not spam, Online transactions Fraud or not Fraud, Tumor Malignant or Benign. Logistic regression transforms its output using the logistic sigmoid function to return a 
probability value. We have also applied this in our datasets and it has a best accuracy score among the two models that we implement(Knn and Svm).

![LogisticRegression Accuracy Score](https://user-images.githubusercontent.com/61632471/126848450-eac539a1-e92a-44c5-9170-f9f91b554452.PNG)



***K Nearest Neighbour Classifier***:<br/>
The k-nearest neighbors (KNN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems.
It's easy to implement and understand, but has a major drawback of becoming significantly slows as the size of that data in use grows.
We have applied this in our datasets and it produces the following accuracy


![Knn Accuracy Score](https://user-images.githubusercontent.com/61632471/126847920-62e66983-4edf-4264-96cc-80965f131628.PNG)


***Support Vector Machine(SVM)***:<br/>
A support vector machine (SVM) is a supervised machine learning model that uses classification algorithms for two-group classification problems. After giving an SVM model
sets of labeled training data for each category, they’re able to categorize new text.Compared to newer algorithms like neural networks, they have two main advantages: 
higher speed and better performance with a limited number of samples (in the thousands). This makes the algorithm very suitable for text classification problems, where 
it’s common to have access to a dataset of at most a couple of thousands of tagged samples and because of this it has a most lowest accuracy score among the two,
and very also turn out to be a very in efficient.

![SVM Accuracy Score](https://user-images.githubusercontent.com/61632471/126848982-56792080-e6fa-4e48-8a36-f386766cb84e.PNG)


***Keggle Score Submission***
-------------------------------------------------------------------------------------------------------------------------
Logistic Regression turns out the best one to train among thre two that we implement. The Reason behind it is that it is easier to implement, interpret, and very 
efficient to train, and more then that It is not only provides a measure of how appropriate a predictor(coefficient size)is, but also its direction of association
(positive or negative).It is also very fast at classyfying the records of datasets that we trained.

![Logistic Regression Keggle Score](https://user-images.githubusercontent.com/61632471/126849366-5e321fd3-ad67-46e0-8d66-b59e384c3ddf.PNG)



