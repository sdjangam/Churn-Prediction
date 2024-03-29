# Abstract

Customers of a biggest international bank decided to leave the banks. 
The bank is investigating a very high rate of customer leaving the bank. 
The dataset contains 10000 records, and we use it to investigate and predict which of the customers are more likely to leave the bank soon. 
The approach here is supervised classification; the classification model to be built on historical data and then used to predict the classes for the current customers to identify the churn. 
The dataset contains 13 features, and also the label column (Exited or not). The best accuracy was obtained with the Naïve Bayes model (82.39%). 
Such churn prediction models could be very useful for applications such as churn prediction in Telecom sector to identify the customers who are switching from current network, and also for Churn prediction in subscription services.

# Motivation for the project

Using the solution to this problem, the bank can easily identify the customers who are willing to exit the bank soon. 
From the larger datasets, the bank can easily identify the churn customers using machine learning approach, thus this can reduce the manual intervention and the cost to the bank. Using machine learning solutions, the bank can save processing time and manual intervention to investigate the complete records. The system can take quicker decisions with statistical models with optimal accuracy metrics.
With the investigation of the customers who are churning soon, the bank has an option to reduce the churn by further investigating the reason of leaving the bank and to convince the customers by providing or improvising the services rendered to them. 
The churn model can be integrated with the call center / business software, so that the proper discounting can be provided to the identified customers. Targeted marketing strategies can be used.
Monitoring of the customer trends and building the alerting mechanism to the business users on a daily / monthly basis.
This problem can be applicable to any industry (for e.g., Telecom) to identify the churn customers within their organizations.

# Problem Definition

The leading international bank wants to investigate the reason for the customers leaving the bank. The bank needs an automated way to predict the customers who are more likely to leave the bank soon. 
The manual intervention to identify the customers who are churning costs the bank in terms of money and the effort to identify the large scale.
So, the bank is looking for a solution with the machine learning algorithms to best fit their historical datasets and to predict the current customers who are likely to churn.
Given: various features of a customer including surname, credit score, geography, gender, age, tenure, balance, number of products, has credit card or not, is active member, estimated salary.
Predict: is he/she likely to leave the bank.
Assumptions
As the customer has provided the limited dataset with limited number of attributes, hence we assume that these are the most important attributes in all their data models.

# Exploratory data analysis and Pre-processing for churn prediction

There are numerous predictive modeling techniques for predicting customer churn. 
These vary in terms of statistical technique (e.g., neural nets versus logistic regression versus survival analysis), and variable selection method (e.g., theory versus stepwise selection). 
Depending on the domain, many domain specific features have been used for churn prediction in the past. For example, for telecom sector, the following features have been used: account length, international plan, voice mail plan, number of voice mail messages, total day minutes used, day calls made, total day charge, total evening minutes, total evening calls, total evening charge, total night minutes, total night calls, total night charge, total international minutes used, total international calls made, total international charge, number customer service calls made. 
Tsai and Lu used two different hybrid models to develop customer churn prediction model. The developed hybrid model is a combination of two artiﬁcial neural networks and the second hybrid model is a combination of self organizing maps and artiﬁcial neural networks. First models are used for data reduction and second models are used for actual classiﬁer. 
Kechadi and Buckley used attribute derivation process to increase the correct prediction rate [2].
Bayesian Belief Network method is tried in a study which is conducted by Kisioglu and Topcu [3]. 
Verbeke et al. increased the accuracy by using two different rules extraction method. These methods were AntMiner+ and ALBA [4]. 
Bock and Poel used two different rotation based ensemble classiﬁers. These are Rotation Forest and Adaboost [5]. 
Yeshwanth et al. suggested a new hybrid model that combines C4.5 decision tree   programming [6]. 
Zhao et al. used one class support vector machine to increase the performance [7]. 
Ghorbani etal. created a new hybrid model by combining neural network, tree models and fuzzy modeling [8]. 
Other recent popular work on churn prediction includes the following.

# Dataset Details

The dataset contains 10000 rows with 14 columns
From where did we get this data?
The source of the dataset is from Kaggle.
What does each attribute stand for?
The dataset contain the columns as listed below 
customerId – represents the customer unique id provided by bank
Surname – sur name of the customer
Credit Score – The credit score for the customer.
Geography – The region where the customer located.
Gender – Male / Female
Age – The customer’s age
Tenure – The duration in years
Balance – The account balance of customer
NumOfProducts – The number of products subscribed / used by the customer.
HasCrCard – The flag to indicate the credit card
IsActiveMember – The activeness of the customer.
EstimatedSalary – The estimated salary for the customer
Exited – Indicates the customer who churned if the value is 1.
Link to the dataset.

 
![image](https://user-images.githubusercontent.com/67232573/111077226-c3c1d680-8515-11eb-8967-fdaac40d2da2.png)


# Exploratory data analysis

![image](https://user-images.githubusercontent.com/67232573/111077258-f370de80-8515-11eb-9a97-712c8e621f73.png)

![image](https://user-images.githubusercontent.com/67232573/111077265-fd92dd00-8515-11eb-89db-ee5374194ab3.png)

![image](https://user-images.githubusercontent.com/67232573/111077284-08e60880-8516-11eb-83ed-f86036dc27fe.png)

![image](https://user-images.githubusercontent.com/67232573/111077291-10a5ad00-8516-11eb-9407-955a939c0800.png)


#  Pre-processing the dataset

Remove outliers.
Handle missing values.
Check for number and types of columns.
We need to create derived attributes for binning the age ranges.
For the attributes, Gender, Geography the categorical variables needs to be convert to factor values for other statistical models other than Decision tree model.
The dummy variable for categorical attributes were created for your reference to look into corresponding mappings for the factorial levels. Later point while applying to the model, these dummy variables were deleted.
In this scenario, as the data is structured and formatted, the text processing was not needed.
Feature importance analysis.

# Pre-processing

Conversion of Categorical values to factor values : Gender, Geography variables were converted from categorical to factors.

![image](https://user-images.githubusercontent.com/67232573/111077382-9295d600-8516-11eb-89cc-e90f54ccf3ac.png)


Creation of bin variables for the Age, Balance, EstimatedSalary variables


![image](https://user-images.githubusercontent.com/67232573/111077410-a80b0000-8516-11eb-9eb2-a23c669762cd.png)


# Technologies/concepts Used

Features identification : 
To determine the importance of variables over the predicted outcome. Here, in our case we need to identify the impact of a variable over Exited column. Ideally, the variables with less importance can be ignored in model building.
Logistic Regression : 
This model can be used to model a binary dependent variable where the two outcomes are 0 and 1.
Naïve Bayes Classifier : 
This model is basically a simple probabilistic classifier considering all the features are independent with each other.
Support Vector Machines (SVM) : 
This model can be used for supervised learning for a classification and regression analysis. SVM model is a representation of points in space and considers as a hyperplane equation to solve the problem.
Decision Trees : 
This model is used to build a decision making using the tree like structure. The corresponding outcome can be derived by traversing over the tree structure.
K-nearest Neighbours(KNN) : 
K-nearest neighbors algorithm is a non-parametric method used for classification and regression problems. The outcome result can be classified by majority of nearest neighbor poins.
Neural Networks : 
This algorithm used to identify the relationships in a set of data through a process as similar to human brain operates. Neural network generates the best possible result without redesign the output criteria.

# Approach

The following classifiers were used to classify the target variable
Logistic regression
Naïve Bayes
Support Vector Machines
Decision Tree
K-Nearest Neighbours (KNN)
Neural Networks
The test data is being passed to the trained models to predict the outcome classes.
Used sklearn package for feature selection / identification and for models.
The accuracy measures and ROC curve was drawn to compare the accuracy metrics among all the models.

# Architecture details of the system

What are the main components in the system?
This Machine learning model can be used as the processing layer to identify the churn out of customers dataset.
The feedback loop can be implemented to continuous training of the model.
What do each of these components do?
The Machine learning model builds the statistical model by fitting the equation over the dataset and can be used to predict the results.
An architecture diagram showing various system modules and their input and output.

# Results

![image](https://user-images.githubusercontent.com/67232573/111077552-56af4080-8517-11eb-9348-d4fc8315ca5e.png)


![image](https://user-images.githubusercontent.com/67232573/111077563-60d13f00-8517-11eb-9c4c-e10cfa813344.png)


# Conclusion

The problem definition is to identify the customers who will churn from the bank.
The Machine learning approach was used to fasten the process in an automated way to identify the churn customers and to minimize the burden of manual processing.
The accuracy metrics and ROC curve for comparing the accuracy metrics among the built models. Identified the customers who will churn.
The churn model can be easily integrate with customer applications to identify the churn and to promote the offers to the customers to maintain the customers without churn.

# Future Work

Can we improve the approach further?
This can be improvised by continuous feedback implementation to identify the future trends and with the additional attributes. 
Can we get better data to design an improved solution?
We can request the additional data from the customer to use the extra features to improvise the solution.
Can we try any other experimental settings like some other pre-processing steps?
The customer datasets might have missing values, those can be handled with imputation techniques as a pre-processing steps and the proper imputation methods and other pre-processing steps can be considered to best improvise the model.
