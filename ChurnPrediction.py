# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from numpy import median
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import matplotlib.lines as mlines
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Change working directory
import os
os.chdir("/Project/ChurnPrediction/")

# Load Dataset
inputPath = "Data/Churn_Modelling.csv"
dataset = pd.read_csv(inputPath, header=0)
dataset.head()

# Exploratory Analysis
dataset.describe()

# Categorical data points exploration
# Gender, Geography are the useful data points, where as surname is of no significance for the model.

dataset.groupby("Gender")["Geography"].count()
dataset.groupby("Geography")["Gender"].count()
# Conversion of categorical values into numerical levels
dataset["Gender1"] = dataset["Gender"]
dataset["Gender"] = pd.Categorical(dataset["Gender"])
dataset["Gender"] = dataset["Gender"].cat.codes
dataset.head()

dataset["Geography1"] = dataset["Geography"]
dataset["Geography"] = pd.Categorical(dataset["Geography"])
dataset["Geography"] = dataset["Geography"].cat.codes
dataset.head()

# Age binning
dataset["AgeBin"] = pd.cut(dataset['Age'], [0, 16, 32,48,64,500])
dataset["AgeBin"] = pd.Categorical(dataset["AgeBin"])
dataset["AgeBin"] = dataset["AgeBin"].cat.codes
dataset.loc[dataset["Age"] > 60].head()

# Binning credit score
dataset['CreditScoreBin'] = pd.cut(dataset['CreditScore'], [0, 450, 550,650,750,900])

dataset["CreditScoreBin"] = pd.Categorical(dataset["CreditScoreBin"])
dataset["CreditScoreBin"] = dataset["CreditScoreBin"].cat.codes
dataset.head()

# Binning Balance
dataset['BalanceBin'] = pd.cut(dataset['Balance'], [-1, 50000, 100000,150000,200000,1000000000000000])

dataset["BalanceBin"] = pd.Categorical(dataset["BalanceBin"])
dataset["BalanceBin"] = dataset["BalanceBin"].cat.codes
dataset.head()

# Binning Estimated Salary
dataset['EstimatedSalaryBin'] = pd.cut(dataset['EstimatedSalary'], [-1, 50000, 100000,150000,200000,1000000000000000])

dataset["EstimatedSalaryBin"] = pd.Categorical(dataset["EstimatedSalaryBin"])
dataset["EstimatedSalaryBin"] = dataset["EstimatedSalaryBin"].cat.codes
dataset.head()

# Box plot
fig, ((a,b,c,d),(e,f,g,h)) = plt.subplots(2,4)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=1, hspace=0.6)

a.set_title("Age")
a.boxplot(dataset["Age"])
b.set_title("CreditScore")
b.boxplot(dataset["CreditScore"])
c.set_title("Tenure")
c.boxplot(dataset["Tenure"])
d.set_title("Balance")
d.boxplot(dataset["Balance"])
e.set_title("NumOfProducts")
e.boxplot(dataset["NumOfProducts"])
f.set_title("HasCrCard")
f.boxplot(dataset["HasCrCard"])
g.set_title("IsActiveMember")
g.boxplot(dataset["IsActiveMember"])
h.set_title("EstimatedSalary")
h.boxplot(dataset["EstimatedSalary"])
plt.show()

# Correlation
dataset.corr()["Exited"]

# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(x_train, y_train)
print(fit.scores_)
d = {'columnName': x_train.columns.values, 'featureScore': fit.scores_}
df = pd.DataFrame(data=d)
print(df.sort_values(['featureScore'], ascending=False))

fig, ((a,b)) = plt.subplots(1,2)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=1, hspace=1)
a.set_title("Data points per Gender")
a.bar(dataset["Gender1"].unique(),dataset.iloc[:,14].value_counts())
b.set_title("Data points per Geography")
b.bar(dataset["Geography1"].unique(),dataset.iloc[:,15].value_counts())
plt.show()

# Remove the non-necessary fields
dataset1 = dataset.copy()
dataset = dataset.drop(["CustomerId"], axis=1)
dataset = dataset.drop(["Gender1"], axis=1)
dataset = dataset.drop(["Geography1"], axis=1)
dataset = dataset.drop(["Age"], axis=1)
dataset = dataset.drop(["CreditScore"], axis=1)
dataset = dataset.drop(["Balance"], axis=1)
dataset = dataset.drop(["EstimatedSalary"], axis=1)
dataset = dataset.drop(["Surname"], axis=1)
dataset = dataset.drop(["RowNumber"], axis=1)

# Random shuffle of records 
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Split data into train and test datasets
test_data_split = 0.2
x_train,x_test , y_train, y_test = train_test_split(dataset.drop(["Exited"],axis=1),dataset["Exited"],test_size = test_data_split)
x_test.describe()

# Check the distribution of Exited in train & Test datasets
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=1, hspace=1)
plt.subplot(1,3,1)
y_train.iloc[:].value_counts().plot(kind = 'bar',title="train Dataset")
plt.subplot(1,3,2)
y_test.iloc[:].value_counts().plot(kind = 'bar',title="test Dataset")
plt.subplot(1,3,3)
dataset.iloc[:,7].value_counts().plot(kind = 'bar',title="Complete Dataset")
plt.show()
# Build model
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)


# Accuracy metrics
#print(Y_pred)
acc_log = round(logreg.score(x_test, y_test), 3)
print("accuracy:", acc_log)

# Prediction
y_pred = logreg.predict(x_test)
# Confusion Matrix
confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
(tn, fp, fn, tp)

# ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()

y_pred_proba = logreg.predict_proba(x_test)[::,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

roc_auc = auc(fpr,tpr)
print(roc_auc)

plt.title("ROC Curve (Logistic Regression)")
plt.plot(fpr, tpr, color='blue',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.show()
# Naive Bayes classification

gnb = GaussianNB()
# Train classifier
gnb.fit(
    x_train,
    y_train
)
y_pred = gnb.predict(x_test)

accuracy_score(y_test,y_pred)

# ROC curve

fpr = dict()
tpr = dict()
roc_auc = dict()

y_pred_proba = gnb.predict_proba(x_test)[::,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

roc_auc = auc(fpr,tpr)
print(roc_auc)

plt.title("ROC Curve (Naive Bayes)")
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.show()
# SVM
model_svm = svm.SVC(probability=True)
model_svm.fit(x_train, y_train)

# Predict SVM
y_pred = model_svm.predict(x_test)
# model_svm.support_vectors_


# ROC curve

fpr = dict()
tpr = dict()
roc_auc = dict()

y_pred_proba = model_svm.predict_proba(x_test)[::,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

roc_auc = auc(fpr,tpr)
print(roc_auc)

plt.title("ROC Curve (Support Vector Machine - SVM)")
plt.plot(fpr, tpr, color='green',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.show()

# ROC comparision among 3 models

fpr = dict()
tpr = dict()
roc_auc = dict()

y_pred_proba_glm = logreg.predict_proba(x_test)[::,1]
y_pred_proba_nb = gnb.predict_proba(x_test)[::,1]
y_pred_proba_svm = model_svm.predict_proba(x_test)[::,1]

fpr[0], tpr[0], thresholds_glm = roc_curve(y_test, y_pred_proba_glm)
fpr[1], tpr[1], thresholds_nb = roc_curve(y_test, y_pred_proba_nb)
fpr[2], tpr[2], thresholds_svm = roc_curve(y_test, y_pred_proba_svm)

roc_auc[0] = auc(fpr[0],tpr[0])
roc_auc[1] = auc(fpr[1],tpr[1])
roc_auc[2] = auc(fpr[2],tpr[2])
print(roc_auc)
colors = ['blue', 'darkorange','green']
titles = ["Logistic","Naive Bayes","SVM"]
plt.title("ROC Curve")
for i in range(3):
    plt.plot(fpr[i], tpr[i], color=colors[i],lw=2, label='ROC curve (area = %0.2f)' % roc_auc[i])
# Labels inside the plot area (ROC Curve)    
blue_line = mlines.Line2D([], [], color='blue', label='Logistic regression')
orange_line = mlines.Line2D([], [], color='darkorange', label='Naive Bayes')
green_line = mlines.Line2D([], [], color='green', label='SVM')
plt.legend(handles=[blue_line, orange_line,green_line])
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.show()