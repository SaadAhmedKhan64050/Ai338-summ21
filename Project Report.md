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


***Keggle Score Submission***:
-------------------------------------------------------------------------------------------------------------------------
Logistic Regression turns out the best one to train among thre two that we implement. The Reason behind it is that it is easier to implement, interpret, and very 
efficient to train, and more then that It is not only provides a measure of how appropriate a predictor(coefficient size)is, but also its direction of association
(positive or negative).It is also very fast at classyfying the records of datasets that we trained.

![Logistic Regression Keggle Score](https://user-images.githubusercontent.com/61632471/126849366-5e321fd3-ad67-46e0-8d66-b59e384c3ddf.PNG)

***How the feature selection affect the prediction and score???***
----------------------------------------------------------------
We only take a take required feeatures to train the models.When we get any dataset, not necessarily every column (feature) is going to have an impact on the output variable. If we add these irrelevant features in the model, it will just make the model worst (Garbage In Garbage Out). This gives rise to the need of doing feature selection.
Let me Elobrate how:<br/>
This is my current accuracy of our best model:<br/>

![image](https://user-images.githubusercontent.com/61632471/126851581-73ae1373-b456-498e-b0d5-0687fe09e498.png) <br/>

Now when I add the extra feature to the dropping coloumn instead of my three relevant datasets
 of each test and train like the image shown below:<br/>
 
 ![image](https://user-images.githubusercontent.com/61632471/126851681-a580e1f5-932e-40f3-8762-698a6a5ca50d.png) <br/>
 
 It will reduce the accuracy and make the model worst because of the extra irrelavant data feature that I convert 
 
 ![image](https://user-images.githubusercontent.com/61632471/126851770-c0dc5d92-b564-4c42-b590-fb05d1d629dd.png)<br/>
 See How the score drop from 0.803 to 0.789

***What we Learned***:
---------------------------------------------------------

In this project we've learned how to train the model, evaluation of those model, conversting and creating the data set ro accurate score. Calculate the score of all three three models and create the dummy values by converting the string values into integar value. One of the biggest learning point was, by K fold cross validation we found the accurate score and how by tweeking we can change the parameters of the models.
