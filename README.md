# Introduction
Here we are trying to predict the overall stock market trend by
sentiment analysis of 'top 25 news headlines' from 2008-08-11 to 2016-07-01.<br>
We built various machine learning models on this data and tried to predict the
trend.<br><br>
The datasets used for this projects are:
1. #### Reddit news
data: 
   Historical news headlines from Reddit News Channel. They are ranked by
the reddit user's votes.
2. #### Stock data: 
   Dow Jones Industrial Average
(DJIA) is used as the stock market data.

Let's import some relevant packages.

```python
# Load in the relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
```

### 1. Data transformation:
Load in our news data. The sentiment scores for the
news is obtained using Stanford NLP Software. 

Here we are adding the sentiment
scores of 5 consecutive days using 'rolling window'. This will be
the basis of
next day stock trend.

```python
# Load in setiment score's dataset
df = pd.read_csv("Sent_scores.csv", index_col=0)

# Add 5 consecutive day's scores
df["PosCount_cv"] = df["PosCount"].rolling(window=5).sum()
df["NegCount_cv"] = df["NegCount"].rolling(window=5).sum()
df["TrustCount_cv"] = df["TrustCount"].rolling(window=5).sum()
df["AngerCount_cv"] = df["AngerCount"].rolling(window=5).sum()
df["AnticipationCount_cv"] = df["AnticipationCount"].rolling(window=5).sum()
df["DisgustCount_cv"] = df["DisgustCount"].rolling(window=5).sum()
df["FearCount_cv"] = df["FearCount"].rolling(window=5).sum()
df["JoyCount_cv"] = df["JoyCount"].rolling(window=5).sum()
df["SadnessCount_cv"] = df["SadnessCount"].rolling(window=5).sum()
df["SurpriseCount_cv"] = df["SurpriseCount"].rolling(window=5).sum()
df.head(5)
```

```python
# Load in stock market data
stock_df = pd.read_csv("DJIA_table.csv", index_col=1)
stock_df = stock_df.iloc[::-1]
stock_df.head(5)
```

Merge the two datasets and drop the irrelevant variables. Also, assign the
binary class to each day stock trend. If today's stock price is larger than
previous day's then assign class '1' otherwise assign class '0'.

```python
# Mearge two datasets
df1 = pd.merge(df, stock_df)

# Drop irrelevant variabls
df2 = df1.drop('Volume', axis=1)
df2 = df2.drop('High', axis=1)
df2 = df2.drop('Low', axis=1)
df2 = df2.drop('Close', axis=1)
#df2 = df2.drop('Date', axis=1)
df2["trend"] = 0

# Assigning class to each day stock trend
for i in range(1,len(df2["Adj Close"])):
    if(df2["Adj Close"].iloc[i]>=df2["Adj Close"].iloc[i-1]):
        df2["trend"].iloc[i]=1
    else:
        df2["trend"].iloc[i]=0

df2 = df2.drop(df2.index[[0]])
df2.head(5)
```

```python
df2 = df2.drop('Adj Close', axis=1)
df2["trend"].head(5)
```

#### Train and Test split:
Split the 70% data as a training set and other 30% as
a test set which will be used for predicting stock market trend.

```python
# train-test split
train = df2.loc[0:1392,:]
test = df2.drop(train.index)

date_train = train["Date"]
date_test = test["Date"]

train = train.drop('Date', axis=1)
test = test.drop('Date',axis=1)

X = train.loc[:, train.columns!="trend"]
y = train.iloc[:,-1]

test_X = test.loc[:, test.columns!="trend"]
test_y = test.iloc[:,-1]

train.head(5)
```

### 2. Machine Learning models
Implementing some machine learning models
including Random forest, KNN, Neural Networks, Naive Bayes, Decision tree, etc.
using scikit-learn library to predict the stock trend.

### Random Forest

```python
clf = RandomForestClassifier(max_depth=2, random_state=0)
#clf = GradientBoostingClassifier(n_estimators=1000, max_depth=10)
#clf = AdaBoostClassifier(n_estimators=100)
#clf = linear_model.LinearRegression()

clf.fit(X,y)
test["predict"] = clf.predict(test_X)
test.head(5)
```

```python
#confusion_matrix
cnf_matrix = confusion_matrix(test["trend"], test["predict"])
np.set_printoptions(precision=2)
class_names = ['1','0']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()
```

```python
print(accuracy_score(test["trend"], test["predict"]))
print(confusion_matrix(test["trend"], test["predict"]))
print(classification_report(test["trend"], test["predict"]))
```

## Feature Importance
As expected, importance of features - 'moving sum
sentiment scores' is higher than daywise sentiment scores!

```python
# feature_importancefrom sklearn import tree
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names= X.columns[indices]
f, ax = plt.subplots(figsize=(11, 9))
plt.title("Feature ranking", fontsize = 20)
plt.bar(range(X.shape[1]), importances[indices],color="b", align="center")
plt.xticks(range(X.shape[1]), feature_names)
plt.xlim([-1, X.shape[1]])
plt.ylabel("importance", fontsize = 18)
plt.xlabel("index of the feature", fontsize = 0.05)
plt.xticks(rotation=90)
plt.show()
```

### Gradient Boosting Classifier

```python
clf = GradientBoostingClassifier(n_estimators=1000, max_depth=10)
#clf = AdaBoostClassifier(n_estimators=100)
#clf = linear_model.LinearRegression()

clf.fit(X,y)
test["predict"] = clf.predict(test_X)
test.head(5)
```

```python
#confusion_matrix
cnf_matrix = confusion_matrix(test["trend"], test["predict"])
np.set_printoptions(precision=2)
class_names = ['1','0']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()

print(accuracy_score(test["trend"], test["predict"]))
print(confusion_matrix(test["trend"], test["predict"]))
print(classification_report(test["trend"], test["predict"]))
```

```python
print(accuracy_score(test["trend"], test["predict"]))
print(confusion_matrix(test["trend"], test["predict"]))
print(classification_report(test["trend"], test["predict"]))
```

### AdaBoost Classifier

```python
#clf = GradientBoostingClassifier(n_estimators=1000, max_depth=10)
clf = AdaBoostClassifier(n_estimators=100)
#clf = linear_model.LinearRegression()

clf.fit(X,y)
test["predict"] = clf.predict(test_X)
test.head(5)
```

```python
print(accuracy_score(test["trend"], test["predict"]))
print(confusion_matrix(test["trend"], test["predict"]))
print(classification_report(test["trend"], test["predict"]))
```

### Neural Network

```python
#clf = GradientBoostingClassifier(n_estimators=1000, max_depth=10)
#clf = AdaBoostClassifier(n_estimators=100)
#clf = linear_model.LinearRegression()
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,10), random_state=1)

clf.fit(X,y)
test["predict"] = clf.predict(test_X)
test.head(5)
```

```python
print(accuracy_score(test["trend"], test["predict"]))
print(confusion_matrix(test["trend"], test["predict"]))
print(classification_report(test["trend"], test["predict"]))
```

### Naive Bayes

```python
# Naive Bayes Algo
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, y)
test["predict"] = clf.predict(test_X)
test.head(5)
```

```python
print(accuracy_score(test["trend"], test["predict"]))
print(confusion_matrix(test["trend"], test["predict"]))
print(classification_report(test["trend"], test["predict"]))
```

### Conclusion:
The maximum accuracy of predicting overall stock market trend is
55%. It implies that sentiment of news headlines have the impact on stock market
trend.

```python

```
