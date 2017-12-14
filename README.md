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

```{.python .input  n=38}
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

```{.python .input  n=39}
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

```{.json .output n=39}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Date</th>\n      <th>PosCount</th>\n      <th>NegCount</th>\n      <th>TrustCount</th>\n      <th>AngerCount</th>\n      <th>AnticipationCount</th>\n      <th>DisgustCount</th>\n      <th>FearCount</th>\n      <th>JoyCount</th>\n      <th>SadnessCount</th>\n      <th>...</th>\n      <th>PosCount_cv</th>\n      <th>NegCount_cv</th>\n      <th>TrustCount_cv</th>\n      <th>AngerCount_cv</th>\n      <th>AnticipationCount_cv</th>\n      <th>DisgustCount_cv</th>\n      <th>FearCount_cv</th>\n      <th>JoyCount_cv</th>\n      <th>SadnessCount_cv</th>\n      <th>SurpriseCount_cv</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1</th>\n      <td>2008-06-08</td>\n      <td>19</td>\n      <td>28</td>\n      <td>12</td>\n      <td>15</td>\n      <td>9</td>\n      <td>5</td>\n      <td>22</td>\n      <td>3</td>\n      <td>10</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>2008-06-09</td>\n      <td>25</td>\n      <td>25</td>\n      <td>20</td>\n      <td>16</td>\n      <td>16</td>\n      <td>3</td>\n      <td>22</td>\n      <td>9</td>\n      <td>9</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>2008-06-10</td>\n      <td>11</td>\n      <td>27</td>\n      <td>12</td>\n      <td>16</td>\n      <td>12</td>\n      <td>9</td>\n      <td>21</td>\n      <td>6</td>\n      <td>12</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>2008-06-11</td>\n      <td>19</td>\n      <td>19</td>\n      <td>15</td>\n      <td>11</td>\n      <td>6</td>\n      <td>9</td>\n      <td>15</td>\n      <td>7</td>\n      <td>6</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>5</th>\n      <td>2008-06-12</td>\n      <td>17</td>\n      <td>24</td>\n      <td>15</td>\n      <td>15</td>\n      <td>8</td>\n      <td>10</td>\n      <td>20</td>\n      <td>6</td>\n      <td>12</td>\n      <td>...</td>\n      <td>91.0</td>\n      <td>123.0</td>\n      <td>74.0</td>\n      <td>73.0</td>\n      <td>51.0</td>\n      <td>36.0</td>\n      <td>100.0</td>\n      <td>31.0</td>\n      <td>49.0</td>\n      <td>18.0</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 21 columns</p>\n</div>",
   "text/plain": "         Date  PosCount  NegCount  TrustCount  AngerCount  AnticipationCount  \\\n1  2008-06-08        19        28          12          15                  9   \n2  2008-06-09        25        25          20          16                 16   \n3  2008-06-10        11        27          12          16                 12   \n4  2008-06-11        19        19          15          11                  6   \n5  2008-06-12        17        24          15          15                  8   \n\n   DisgustCount  FearCount  JoyCount  SadnessCount        ...         \\\n1             5         22         3            10        ...          \n2             3         22         9             9        ...          \n3             9         21         6            12        ...          \n4             9         15         7             6        ...          \n5            10         20         6            12        ...          \n\n   PosCount_cv  NegCount_cv  TrustCount_cv  AngerCount_cv  \\\n1          NaN          NaN            NaN            NaN   \n2          NaN          NaN            NaN            NaN   \n3          NaN          NaN            NaN            NaN   \n4          NaN          NaN            NaN            NaN   \n5         91.0        123.0           74.0           73.0   \n\n   AnticipationCount_cv  DisgustCount_cv  FearCount_cv  JoyCount_cv  \\\n1                   NaN              NaN           NaN          NaN   \n2                   NaN              NaN           NaN          NaN   \n3                   NaN              NaN           NaN          NaN   \n4                   NaN              NaN           NaN          NaN   \n5                  51.0             36.0         100.0         31.0   \n\n   SadnessCount_cv  SurpriseCount_cv  \n1              NaN               NaN  \n2              NaN               NaN  \n3              NaN               NaN  \n4              NaN               NaN  \n5             49.0              18.0  \n\n[5 rows x 21 columns]"
  },
  "execution_count": 39,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=30}
# Load in stock market data
stock_df = pd.read_csv("DJIA_table.csv", index_col=1)
stock_df = stock_df.iloc[::-1]
stock_df.head(5)
```

```{.json .output n=30}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Date</th>\n      <th>High</th>\n      <th>Low</th>\n      <th>Close</th>\n      <th>Volume</th>\n      <th>Adj Close</th>\n    </tr>\n    <tr>\n      <th>Open</th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>11432.089844</th>\n      <td>2008-08-08</td>\n      <td>11759.959961</td>\n      <td>11388.040039</td>\n      <td>11734.320312</td>\n      <td>212830000</td>\n      <td>11734.320312</td>\n    </tr>\n    <tr>\n      <th>11729.669922</th>\n      <td>2008-08-11</td>\n      <td>11867.110352</td>\n      <td>11675.530273</td>\n      <td>11782.349609</td>\n      <td>183190000</td>\n      <td>11782.349609</td>\n    </tr>\n    <tr>\n      <th>11781.700195</th>\n      <td>2008-08-12</td>\n      <td>11782.349609</td>\n      <td>11601.519531</td>\n      <td>11642.469727</td>\n      <td>173590000</td>\n      <td>11642.469727</td>\n    </tr>\n    <tr>\n      <th>11632.809570</th>\n      <td>2008-08-13</td>\n      <td>11633.780273</td>\n      <td>11453.339844</td>\n      <td>11532.959961</td>\n      <td>182550000</td>\n      <td>11532.959961</td>\n    </tr>\n    <tr>\n      <th>11532.070312</th>\n      <td>2008-08-14</td>\n      <td>11718.280273</td>\n      <td>11450.889648</td>\n      <td>11615.929688</td>\n      <td>159790000</td>\n      <td>11615.929688</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                    Date          High           Low         Close     Volume  \\\nOpen                                                                            \n11432.089844  2008-08-08  11759.959961  11388.040039  11734.320312  212830000   \n11729.669922  2008-08-11  11867.110352  11675.530273  11782.349609  183190000   \n11781.700195  2008-08-12  11782.349609  11601.519531  11642.469727  173590000   \n11632.809570  2008-08-13  11633.780273  11453.339844  11532.959961  182550000   \n11532.070312  2008-08-14  11718.280273  11450.889648  11615.929688  159790000   \n\n                 Adj Close  \nOpen                        \n11432.089844  11734.320312  \n11729.669922  11782.349609  \n11781.700195  11642.469727  \n11632.809570  11532.959961  \n11532.070312  11615.929688  "
  },
  "execution_count": 30,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Merge the two datasets and drop the irrelevant variables. Also, assign the
binary class to each day stock trend. If today's stock price is larger than
previous day's then assign class '1' otherwise assign class '0'.

```{.python .input  n=40}
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

```{.json .output n=40}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Date</th>\n      <th>PosCount</th>\n      <th>NegCount</th>\n      <th>TrustCount</th>\n      <th>AngerCount</th>\n      <th>AnticipationCount</th>\n      <th>DisgustCount</th>\n      <th>FearCount</th>\n      <th>JoyCount</th>\n      <th>SadnessCount</th>\n      <th>...</th>\n      <th>TrustCount_cv</th>\n      <th>AngerCount_cv</th>\n      <th>AnticipationCount_cv</th>\n      <th>DisgustCount_cv</th>\n      <th>FearCount_cv</th>\n      <th>JoyCount_cv</th>\n      <th>SadnessCount_cv</th>\n      <th>SurpriseCount_cv</th>\n      <th>Adj Close</th>\n      <th>trend</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1</th>\n      <td>2008-08-11</td>\n      <td>12</td>\n      <td>15</td>\n      <td>7</td>\n      <td>8</td>\n      <td>7</td>\n      <td>3</td>\n      <td>15</td>\n      <td>5</td>\n      <td>6</td>\n      <td>...</td>\n      <td>51.0</td>\n      <td>48.0</td>\n      <td>33.0</td>\n      <td>20.0</td>\n      <td>87.0</td>\n      <td>19.0</td>\n      <td>37.0</td>\n      <td>22.0</td>\n      <td>11782.349609</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>2008-08-12</td>\n      <td>12</td>\n      <td>21</td>\n      <td>6</td>\n      <td>10</td>\n      <td>3</td>\n      <td>1</td>\n      <td>16</td>\n      <td>4</td>\n      <td>8</td>\n      <td>...</td>\n      <td>41.0</td>\n      <td>47.0</td>\n      <td>28.0</td>\n      <td>15.0</td>\n      <td>84.0</td>\n      <td>20.0</td>\n      <td>37.0</td>\n      <td>20.0</td>\n      <td>11642.469727</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>2008-08-13</td>\n      <td>21</td>\n      <td>19</td>\n      <td>19</td>\n      <td>12</td>\n      <td>5</td>\n      <td>2</td>\n      <td>16</td>\n      <td>6</td>\n      <td>8</td>\n      <td>...</td>\n      <td>52.0</td>\n      <td>45.0</td>\n      <td>26.0</td>\n      <td>9.0</td>\n      <td>78.0</td>\n      <td>24.0</td>\n      <td>36.0</td>\n      <td>19.0</td>\n      <td>11532.959961</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>2008-08-14</td>\n      <td>22</td>\n      <td>21</td>\n      <td>16</td>\n      <td>12</td>\n      <td>15</td>\n      <td>3</td>\n      <td>17</td>\n      <td>9</td>\n      <td>10</td>\n      <td>...</td>\n      <td>59.0</td>\n      <td>52.0</td>\n      <td>38.0</td>\n      <td>11.0</td>\n      <td>81.0</td>\n      <td>29.0</td>\n      <td>42.0</td>\n      <td>23.0</td>\n      <td>11615.929688</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>5</th>\n      <td>2008-08-15</td>\n      <td>15</td>\n      <td>23</td>\n      <td>14</td>\n      <td>14</td>\n      <td>4</td>\n      <td>3</td>\n      <td>20</td>\n      <td>8</td>\n      <td>8</td>\n      <td>...</td>\n      <td>62.0</td>\n      <td>56.0</td>\n      <td>34.0</td>\n      <td>12.0</td>\n      <td>84.0</td>\n      <td>32.0</td>\n      <td>40.0</td>\n      <td>22.0</td>\n      <td>11659.900391</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 23 columns</p>\n</div>",
   "text/plain": "         Date  PosCount  NegCount  TrustCount  AngerCount  AnticipationCount  \\\n1  2008-08-11        12        15           7           8                  7   \n2  2008-08-12        12        21           6          10                  3   \n3  2008-08-13        21        19          19          12                  5   \n4  2008-08-14        22        21          16          12                 15   \n5  2008-08-15        15        23          14          14                  4   \n\n   DisgustCount  FearCount  JoyCount  SadnessCount  ...    TrustCount_cv  \\\n1             3         15         5             6  ...             51.0   \n2             1         16         4             8  ...             41.0   \n3             2         16         6             8  ...             52.0   \n4             3         17         9            10  ...             59.0   \n5             3         20         8             8  ...             62.0   \n\n   AngerCount_cv  AnticipationCount_cv  DisgustCount_cv  FearCount_cv  \\\n1           48.0                  33.0             20.0          87.0   \n2           47.0                  28.0             15.0          84.0   \n3           45.0                  26.0              9.0          78.0   \n4           52.0                  38.0             11.0          81.0   \n5           56.0                  34.0             12.0          84.0   \n\n   JoyCount_cv  SadnessCount_cv  SurpriseCount_cv     Adj Close  trend  \n1         19.0             37.0              22.0  11782.349609      1  \n2         20.0             37.0              20.0  11642.469727      0  \n3         24.0             36.0              19.0  11532.959961      0  \n4         29.0             42.0              23.0  11615.929688      1  \n5         32.0             40.0              22.0  11659.900391      1  \n\n[5 rows x 23 columns]"
  },
  "execution_count": 40,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=41}
df2 = df2.drop('Adj Close', axis=1)
df2["trend"].head(5)
```

```{.json .output n=41}
[
 {
  "data": {
   "text/plain": "1    1\n2    0\n3    0\n4    1\n5    1\nName: trend, dtype: int64"
  },
  "execution_count": 41,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

#### Train and Test split:
Split the 70% data as a training set and other 30% as
a test set which will be used for predicting stock market trend.

```{.python .input  n=42}
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

```{.json .output n=42}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>PosCount</th>\n      <th>NegCount</th>\n      <th>TrustCount</th>\n      <th>AngerCount</th>\n      <th>AnticipationCount</th>\n      <th>DisgustCount</th>\n      <th>FearCount</th>\n      <th>JoyCount</th>\n      <th>SadnessCount</th>\n      <th>SurpriseCount</th>\n      <th>...</th>\n      <th>NegCount_cv</th>\n      <th>TrustCount_cv</th>\n      <th>AngerCount_cv</th>\n      <th>AnticipationCount_cv</th>\n      <th>DisgustCount_cv</th>\n      <th>FearCount_cv</th>\n      <th>JoyCount_cv</th>\n      <th>SadnessCount_cv</th>\n      <th>SurpriseCount_cv</th>\n      <th>trend</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1</th>\n      <td>12</td>\n      <td>15</td>\n      <td>7</td>\n      <td>8</td>\n      <td>7</td>\n      <td>3</td>\n      <td>15</td>\n      <td>5</td>\n      <td>6</td>\n      <td>5</td>\n      <td>...</td>\n      <td>99.0</td>\n      <td>51.0</td>\n      <td>48.0</td>\n      <td>33.0</td>\n      <td>20.0</td>\n      <td>87.0</td>\n      <td>19.0</td>\n      <td>37.0</td>\n      <td>22.0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>12</td>\n      <td>21</td>\n      <td>6</td>\n      <td>10</td>\n      <td>3</td>\n      <td>1</td>\n      <td>16</td>\n      <td>4</td>\n      <td>8</td>\n      <td>3</td>\n      <td>...</td>\n      <td>95.0</td>\n      <td>41.0</td>\n      <td>47.0</td>\n      <td>28.0</td>\n      <td>15.0</td>\n      <td>84.0</td>\n      <td>20.0</td>\n      <td>37.0</td>\n      <td>20.0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>21</td>\n      <td>19</td>\n      <td>19</td>\n      <td>12</td>\n      <td>5</td>\n      <td>2</td>\n      <td>16</td>\n      <td>6</td>\n      <td>8</td>\n      <td>3</td>\n      <td>...</td>\n      <td>89.0</td>\n      <td>52.0</td>\n      <td>45.0</td>\n      <td>26.0</td>\n      <td>9.0</td>\n      <td>78.0</td>\n      <td>24.0</td>\n      <td>36.0</td>\n      <td>19.0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>22</td>\n      <td>21</td>\n      <td>16</td>\n      <td>12</td>\n      <td>15</td>\n      <td>3</td>\n      <td>17</td>\n      <td>9</td>\n      <td>10</td>\n      <td>8</td>\n      <td>...</td>\n      <td>96.0</td>\n      <td>59.0</td>\n      <td>52.0</td>\n      <td>38.0</td>\n      <td>11.0</td>\n      <td>81.0</td>\n      <td>29.0</td>\n      <td>42.0</td>\n      <td>23.0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>5</th>\n      <td>15</td>\n      <td>23</td>\n      <td>14</td>\n      <td>14</td>\n      <td>4</td>\n      <td>3</td>\n      <td>20</td>\n      <td>8</td>\n      <td>8</td>\n      <td>3</td>\n      <td>...</td>\n      <td>99.0</td>\n      <td>62.0</td>\n      <td>56.0</td>\n      <td>34.0</td>\n      <td>12.0</td>\n      <td>84.0</td>\n      <td>32.0</td>\n      <td>40.0</td>\n      <td>22.0</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 21 columns</p>\n</div>",
   "text/plain": "   PosCount  NegCount  TrustCount  AngerCount  AnticipationCount  \\\n1        12        15           7           8                  7   \n2        12        21           6          10                  3   \n3        21        19          19          12                  5   \n4        22        21          16          12                 15   \n5        15        23          14          14                  4   \n\n   DisgustCount  FearCount  JoyCount  SadnessCount  SurpriseCount  ...    \\\n1             3         15         5             6              5  ...     \n2             1         16         4             8              3  ...     \n3             2         16         6             8              3  ...     \n4             3         17         9            10              8  ...     \n5             3         20         8             8              3  ...     \n\n   NegCount_cv  TrustCount_cv  AngerCount_cv  AnticipationCount_cv  \\\n1         99.0           51.0           48.0                  33.0   \n2         95.0           41.0           47.0                  28.0   \n3         89.0           52.0           45.0                  26.0   \n4         96.0           59.0           52.0                  38.0   \n5         99.0           62.0           56.0                  34.0   \n\n   DisgustCount_cv  FearCount_cv  JoyCount_cv  SadnessCount_cv  \\\n1             20.0          87.0         19.0             37.0   \n2             15.0          84.0         20.0             37.0   \n3              9.0          78.0         24.0             36.0   \n4             11.0          81.0         29.0             42.0   \n5             12.0          84.0         32.0             40.0   \n\n   SurpriseCount_cv  trend  \n1              22.0      1  \n2              20.0      0  \n3              19.0      0  \n4              23.0      1  \n5              22.0      1  \n\n[5 rows x 21 columns]"
  },
  "execution_count": 42,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### 2. Machine Learning models
Implementing some machine learning models
including Random forest, KNN, Neural Networks, Naive Bayes, Decision tree, etc.
using scikit-learn library to predict the stock trend.

### Random Forest

```{.python .input  n=56}
clf = RandomForestClassifier(max_depth=2, random_state=0)
#clf = GradientBoostingClassifier(n_estimators=1000, max_depth=10)
#clf = AdaBoostClassifier(n_estimators=100)
#clf = linear_model.LinearRegression()

clf.fit(X,y)
test["predict"] = clf.predict(test_X)
test.head(5)
```

```{.json .output n=56}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>PosCount</th>\n      <th>NegCount</th>\n      <th>TrustCount</th>\n      <th>AngerCount</th>\n      <th>AnticipationCount</th>\n      <th>DisgustCount</th>\n      <th>FearCount</th>\n      <th>JoyCount</th>\n      <th>SadnessCount</th>\n      <th>SurpriseCount</th>\n      <th>...</th>\n      <th>TrustCount_cv</th>\n      <th>AngerCount_cv</th>\n      <th>AnticipationCount_cv</th>\n      <th>DisgustCount_cv</th>\n      <th>FearCount_cv</th>\n      <th>JoyCount_cv</th>\n      <th>SadnessCount_cv</th>\n      <th>SurpriseCount_cv</th>\n      <th>trend</th>\n      <th>predict</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1393</th>\n      <td>30</td>\n      <td>35</td>\n      <td>27</td>\n      <td>20</td>\n      <td>9</td>\n      <td>10</td>\n      <td>31</td>\n      <td>6</td>\n      <td>17</td>\n      <td>9</td>\n      <td>...</td>\n      <td>99.0</td>\n      <td>82.0</td>\n      <td>46.0</td>\n      <td>44.0</td>\n      <td>125.0</td>\n      <td>27.0</td>\n      <td>70.0</td>\n      <td>29.0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1394</th>\n      <td>25</td>\n      <td>33</td>\n      <td>20</td>\n      <td>20</td>\n      <td>12</td>\n      <td>5</td>\n      <td>29</td>\n      <td>10</td>\n      <td>14</td>\n      <td>3</td>\n      <td>...</td>\n      <td>107.0</td>\n      <td>77.0</td>\n      <td>54.0</td>\n      <td>38.0</td>\n      <td>112.0</td>\n      <td>41.0</td>\n      <td>63.0</td>\n      <td>30.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1395</th>\n      <td>23</td>\n      <td>25</td>\n      <td>23</td>\n      <td>13</td>\n      <td>12</td>\n      <td>5</td>\n      <td>11</td>\n      <td>10</td>\n      <td>10</td>\n      <td>7</td>\n      <td>...</td>\n      <td>109.0</td>\n      <td>76.0</td>\n      <td>57.0</td>\n      <td>36.0</td>\n      <td>98.0</td>\n      <td>42.0</td>\n      <td>59.0</td>\n      <td>30.0</td>\n      <td>0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1396</th>\n      <td>23</td>\n      <td>22</td>\n      <td>10</td>\n      <td>9</td>\n      <td>5</td>\n      <td>6</td>\n      <td>15</td>\n      <td>4</td>\n      <td>11</td>\n      <td>2</td>\n      <td>...</td>\n      <td>92.0</td>\n      <td>65.0</td>\n      <td>53.0</td>\n      <td>32.0</td>\n      <td>82.0</td>\n      <td>40.0</td>\n      <td>53.0</td>\n      <td>23.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1397</th>\n      <td>27</td>\n      <td>22</td>\n      <td>19</td>\n      <td>13</td>\n      <td>11</td>\n      <td>3</td>\n      <td>18</td>\n      <td>8</td>\n      <td>9</td>\n      <td>4</td>\n      <td>...</td>\n      <td>85.0</td>\n      <td>68.0</td>\n      <td>47.0</td>\n      <td>27.0</td>\n      <td>88.0</td>\n      <td>36.0</td>\n      <td>52.0</td>\n      <td>19.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 22 columns</p>\n</div>",
   "text/plain": "      PosCount  NegCount  TrustCount  AngerCount  AnticipationCount  \\\n1393        30        35          27          20                  9   \n1394        25        33          20          20                 12   \n1395        23        25          23          13                 12   \n1396        23        22          10           9                  5   \n1397        27        22          19          13                 11   \n\n      DisgustCount  FearCount  JoyCount  SadnessCount  SurpriseCount   ...     \\\n1393            10         31         6            17              9   ...      \n1394             5         29        10            14              3   ...      \n1395             5         11        10            10              7   ...      \n1396             6         15         4            11              2   ...      \n1397             3         18         8             9              4   ...      \n\n      TrustCount_cv  AngerCount_cv  AnticipationCount_cv  DisgustCount_cv  \\\n1393           99.0           82.0                  46.0             44.0   \n1394          107.0           77.0                  54.0             38.0   \n1395          109.0           76.0                  57.0             36.0   \n1396           92.0           65.0                  53.0             32.0   \n1397           85.0           68.0                  47.0             27.0   \n\n      FearCount_cv  JoyCount_cv  SadnessCount_cv  SurpriseCount_cv  trend  \\\n1393         125.0         27.0             70.0              29.0      0   \n1394         112.0         41.0             63.0              30.0      1   \n1395          98.0         42.0             59.0              30.0      0   \n1396          82.0         40.0             53.0              23.0      1   \n1397          88.0         36.0             52.0              19.0      1   \n\n      predict  \n1393        0  \n1394        1  \n1395        1  \n1396        1  \n1397        1  \n\n[5 rows x 22 columns]"
  },
  "execution_count": 56,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=57}
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

```{.json .output n=57}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Confusion matrix\n[[ 20 258]\n [ 24 294]]\n"
 },
 {
  "data": {
   "image/png": "iVBORw0KGgoAAAANSUhEUgAAAUsAAAEmCAYAAADr3bIaAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAHzZJREFUeJzt3Xm8VXW9xvHPA0cRBUcUURkccKRkUDRTL4VzGmTlRGZl\nzpmmWTjcqw2k5phlmWZpDii38kpqqWFdwxkVFRARBS4gsxOgIMP3/rEWukXO3usc9j7rrHOe932t\nF3uv8buh+/hbv/VbaykiMDOz8trkXYCZWRE4LM3MMnBYmpll4LA0M8vAYWlmloHD0swsA4dlKyKp\nvaS/SnpH0n+vxX6GSHqomrXlRdJ+kl7Juw5r/uRxls2PpOOAc4CdgYXAWGBYRIxey/0eD5wJ7BMR\ny9e60GZOUgA9I2Jy3rVY8bll2cxIOge4FvgZ0BnoBlwPfLEKu+8OTGoNQZmFpLq8a7ACiQhPzWQC\nNgIWAV8ts047kjB9I52uBdqlywYAM4BzgbnALOCb6bIfAR8Ay9JjnAhcAtxesu8eQAB16fdvAK+T\ntG6nAENK5o8u2W4f4BngnfTPfUqW/Qv4CfBYup+HgE71/LZV9f+gpP7BwGHAJOBN4IKS9fsDTwBv\np+v+Clg3XfZo+lsWp7/36JL9/xCYDdy2al66zfbpMfqm37cC5gED8v7fhqf8J7csm5fPAOsB95RZ\n50Jgb6A3sDtJYFxUsnxLktDdmiQQr5e0SURcTNJavTsiOkTEzeUKkbQBcB1waER0JAnEsWtYb1Pg\n/nTdzYCrgfslbVay2nHAN4EtgHWB75c59JYkfwdbA/8F3AR8DegH7Af8p6Rt03VXAN8DOpH83Q0E\nTgeIiP3TdXZPf+/dJfvflKSVfXLpgSPiNZIgvV3S+sAfgFsj4l9l6rVWwmHZvGwGzI/yp8lDgB9H\nxNyImEfSYjy+ZPmydPmyiHiApFW1UyPrWQn0ktQ+ImZFxPg1rPMF4NWIuC0ilkfEcGAicETJOn+I\niEkR8T4wgiTo67OMpH92GXAXSRD+IiIWpsefQPIfCSLi2Yh4Mj3uVOC3wH9k+E0XR8TStJ6PiYib\ngMnAU0AXkv84mTksm5kFQKcKfWlbAdNKvk9L5324j9XC9j2gQ0MLiYjFJKeupwKzJN0vaecM9ayq\naeuS77MbUM+CiFiRfl4VZnNKlr+/antJO0q6T9JsSe+StJw7ldk3wLyIWFJhnZuAXsAvI2JphXWt\nlXBYNi9PAEtJ+unq8wbJKeQq3dJ5jbEYWL/k+5alCyPiwYg4kKSFNZEkRCrVs6qmmY2sqSF+Q1JX\nz4jYELgAUIVtyg7/kNSBpB/4ZuCStJvBzGHZnETEOyT9dNdLGixpfUnrSDpU0s/T1YYDF0naXFKn\ndP3bG3nIscD+krpJ2gg4f9UCSZ0lDUr7LpeSnM6vXMM+HgB2lHScpDpJRwO7Avc1sqaG6Ai8CyxK\nW72nrbZ8DrBdA/f5C2BMRHybpC/2hrWu0loEh2UzExFXkYyxvIjkSux04DvA/6Sr/BQYA7wIvAQ8\nl85rzLEeBu5O9/UsHw+4Nmkdb5BcIf4PPhlGRMQC4HCSK/ALSK5kHx4R8xtTUwN9n+Ti0UKSVu/d\nqy2/BLhV0tuSjqq0M0mDgEP46HeeA/SVNKRqFVtheVC6mVkGblmamWXgsDQzy8BhaWaWgcPSzCyD\nZvUggU6dOkX37j3yLsOq5PnXm+KCuDWFeG8BsXRhpTGsDdJ2w+4Ryz9xE1X9Nbw/78GIOKSaNTRE\nswrL7t178NhTY/Iuw6pkk6PL3n5uBbL0kZ9UfZ+x/H3a7VRxRNeHloy9vtLdWTXVrMLSzFoTgYrT\nE+iwNLN8CFBVz+xrymFpZvlxy9LMrBJBm7Z5F5GZw9LM8uPTcDOzCoRPw83MKpNblmZmmbhlaWaW\ngVuWZmaVeFC6mVllHpRuZpaRW5ZmZpUI2npQuplZeR5naWaWkfsszcwq8dVwM7Ns3LI0M8vALUsz\nswrke8PNzLJxy9LMLAO3LM3MKvHVcDOzyoRfK2FmVplblmZm2bjP0swsA7cszcwycMvSzKwCuc/S\nzCwbtyzNzCqTw9LMrLzkFTwOSzOz8iTUpjhhWZzeVTNrcSRlnirsp6ukf0qaIGm8pLPS+ZdImilp\nbDodVrLN+ZImS3pF0sGVanXL0sxyU8XT8OXAuRHxnKSOwLOSHk6XXRMRV6523F2BY4DdgK2Af0ja\nMSJW1HcAtyzNLDfVallGxKyIeC79vBB4Gdi6zCaDgLsiYmlETAEmA/3LHcNhaWb5UAOnrLuVegB9\ngKfSWWdKelHS7yVtks7bGphestkMyoerw9LM8iGytyrTlmUnSWNKppM/sU+pA/Bn4OyIeBf4DbAd\n0BuYBVzV2HrdZ2lmuWlgn+X8iNijzL7WIQnKOyLiLwARMadk+U3AfenXmUDXks23SefVyy1LM8tN\nFa+GC7gZeDkiri6Z36VktS8B49LPI4FjJLWTtC3QE3i63DHcsjSz3FTxavhngeOBlySNTeddABwr\nqTcQwFTgFICIGC9pBDCB5Er6GeWuhIPD0szy0sALN+VExOh69vZAmW2GAcOyHsNhaWa5EKJNm+L0\nBDoszSw3vjfczCyL4mSlw9LMciK3LM3MMnFYmpll4LA0M6tg1e2OReGwNLP8FCcrHZa1NH36dL79\nza8zd+4cJPGtE0/mO989izfffJPjjzuaadOm0r17D24fPoJNNtmk8g6tyW2z2Qb87rv7s8VG7Qng\n9w+/wvX3j+fCo/rwrQN2Yt67SwC4+M4xPPjcDOrait+cth+9t9uMurZtuONfr3LlPS/m+yOaK1/g\nsVXq6uq47OdX0advXxYuXMg+e/Vj4AEHctsfb2HA5wdy3g+GcsXPL+PKn1/GsEsvz7tcW4PlK1Yy\n9JanGTtlAR3WW4fHrxjEqBeS5y388r5xXDty3MfW//JntqXdOm3Z85x7aL9uW57/xZcZMfp1/m/e\nojzKb/aKFJbFGT5fQF26dKFP374AdOzYkZ133oU33pjJfX+9l68dfwIAXzv+BP468n/yLNPKmP32\n+4ydsgCARUuWMXHG22y16fr1rh/A+uvV0baNaL9uHR8sX8nC9z9oomqLR22Uecqbw7KJTJs6lbFj\nn2fP/nsxd84cunRJHoay5ZZbMnfOnApbW3PQbfMO9N52M555dR4Apx22G09f/SVuOH0/Nt5gXQD+\n8sQU3luynCm/O5ZJvz2aa0e+xFuLHJb1qdZTh5pCzcIyfSrxXEnjKq/dsi1atIhjj/oyV1x1LRtu\nuOHHljWX/yFYeRusV8fw8wZy3h+eZOH7y7jpwZfZ5fQR7HXuPcx++z0uO2EvAPbcYXNWrFzJdicN\nZ5fTRnDWEb3o0bljztU3Tw0Jyubw/yO1bFneAhxSw/0XwrJlyzj2qC9z9LFDGPylIwHYonNnZs2a\nBcCsWbPYfIst8izRKqhrK4afN5C7//0a9z41DYC57yxh5cogIrnos0fPzQE4ar/teWjsTJavCOa9\nu4QnJs6l3/ad8iy/WXNYAhHxKPBmrfZfBBHBqSedyE4778JZ3zvnw/lfOPyL3H7brQDcftutHH7E\noLxKtAxuOH0/XpnxNtf99aOTpC03bv/h50F7dWfC/70FwIz5ixnQK+liWb9dHf133JxXZr7dtAUX\nSJHCMver4el7NE4G6NqtW87VVNfjjz3GnXfcRq9en2Kvfr0B+NFPf8b3fzCUrx17FLf+4Wa6devO\n7cNH5Fyp1WefnTszZEBPXpr2Jk9eORhIhgkdte/2fLrHpgQwbe5CzrzhMQBu+PsEbjxjf5699kgE\n3PbPVxk37a38fkBzl38GZpZ7WEbEjcCNAP367RE5l1NVn913X95ftuaf9LeHRjVxNdYYj0+cQ/sv\n3/yJ+Q8+N2ON6y9espwhVz1S67JajObQYswq97A0s1bKg9LNzCoTUKCsrOnQoeHAE8BOkmZIOrFW\nxzKzIhJt2mSf8lazlmVEHFurfZtZy+DTcDOzSlSs03CHpZnlQtAsTq+zcliaWW7csjQzy8B9lmZm\nlbjP0syssmScZXHS0mFpZjlpHg/IyMphaWa5KVBWOizNLCfy0CEzs4rcZ2lmllGBstJhaWb5ccvS\nzCyDAmWlX4VrZjlR9d7BI6mrpH9KmiBpvKSz0vmbSnpY0qvpn5uUbHO+pMmSXpF0cKVyHZZmlotV\nD//NOlWwHDg3InYF9gbOkLQrMBQYFRE9gVHpd9JlxwC7kbyF9teS2pY7gMPSzHJSvfeGR8SsiHgu\n/bwQeBnYGhgE3JqudiswOP08CLgrIpZGxBRgMtC/3DHcZ2lmuWlgn2UnSWNKvt+YvvBwtX2qB9AH\neAroHBGz0kWzgc7p562BJ0s2m5HOq5fD0szy0fBB6fMjYo+yu5Q6AH8Gzo6Id0tbpBERkhr9BlmH\npZnlotqD0iWtQxKUd0TEX9LZcyR1iYhZkroAc9P5M4GuJZtvk86rl/sszSw3VbwaLuBm4OWIuLpk\n0UjghPTzCcC9JfOPkdRO0rZAT+Dpcsdwy9LMclPFhuVngeOBlySNTeddAFwGjEjfLjsNOAogIsZL\nGgFMILmSfkZErCh3AIelmeWmWqfhETGa5Mx+TQbWs80wYFjWYzgszSwfflK6mVll8sN/zcyyKVBW\nOizNLD9tCpSWDkszy02BstJhaWb5kKCtXythZlZZi7jAI2nDchtGxLvVL8fMWpMCZWXZluV4IPj4\nQM9V3wPoVsO6zKyFE8nwoaKoNywjomt9y8zMqqFAXZbZHqQh6RhJF6Sft5HUr7ZlmVmL14CHaDSH\nvs2KYSnpV8DnSG5SB3gPuKGWRZlZ61DF10rUXJar4ftERF9JzwNExJuS1q1xXWbWwomWNyh9maQ2\nJBd1kLQZsLKmVZlZq1CgrMzUZ3k9ydOHN5f0I2A0cHlNqzKzVqFIfZYVW5YR8UdJzwIHpLO+GhHj\naluWmbV0LfUOnrbAMpJTcb+KwsyqojhRme1q+IXAcGArkpf63Cnp/FoXZmYtX4s6DQe+DvSJiPcA\nJA0DngcurWVhZtayJVfD864iuyxhOWu19erSeWZmjddMWoxZlXuQxjUkfZRvAuMlPZh+Pwh4pmnK\nM7OWrEBZWbZlueqK93jg/pL5T9auHDNrTVpEyzIibm7KQsysdWlxfZaStid5t+6uwHqr5kfEjjWs\ny8xagSK1LLOMmbwF+APJfwgOBUYAd9ewJjNrBSRoK2We8pYlLNePiAcBIuK1iLiIJDTNzNZKS3vq\n0NL0QRqvSToVmAl0rG1ZZtYaFOk0PEtYfg/YAPguSd/lRsC3almUmbUOBcrKTA/SeCr9uJCPHgBs\nZrZWhFrG8ywl3UP6DMs1iYgja1KRmbUOzaQvMqtyLctfNVkVqQBWrKw3n61oXn8+7wqsWpa+V5Pd\ntog+y4gY1ZSFmFnrU6TnPWZ9nqWZWVWJYrUsixTsZtbCtFH2qRJJv5c0V9K4knmXSJopaWw6HVay\n7HxJkyW9IungSvvP3LKU1C4ilmZd38ysnBq8VuIWkmstf1xt/jURceXHj61dgWOA3UgebP4PSTtG\nxIr6dp7lSen9Jb0EvJp+313SLxv0E8zM1qCaLcuIeJTkkZJZDALuioilETEFmAz0L1trhp1eBxwO\nLEgLegH4XMaCzMzq1cDbHTtJGlMynZzxMGdKejE9Td8knbc1ML1knRnpvHplOQ1vExHTVuuIrbep\namaWRfKItgadhs+PiD0aeJjfAD8hGZn4E+AqGnkHYpawnC6pPxCS2gJnApMaczAzs1K1vsIcEXNW\nfZZ0E3Bf+nUm0LVk1W3SefXKUutpwDlAN2AOsHc6z8xsrdT6qUOSupR8/RIfvQFiJHCMpHaStgV6\nAk+X21eWe8Pnklw1MjOrGqm694ZLGg4MIOnbnAFcDAyQ1JvkNHwqcApARIyXNAKYACwHzih3JRyy\nPSn9JtZwj3hEZO1cNTNbo2qOSY+IY9cwu97X40TEMJInqWWSpc/yHyWf1yNpyk6vZ10zs8xa1Dt4\nIuJjr5CQdBswumYVmVmrIKo+KL2mGnNv+LZA52oXYmatTMbB5s1Flj7Lt/ioz7INyQj5obUsysxa\nB1GctCwblkpGou/OR+OPVkaEHzhpZmutaO8NLzvOMg3GByJiRTo5KM2saqp5b3jNa82wzlhJfWpe\niZm1OpIyT3kr9w6euohYDvQBnpH0GrCYpPUcEdG3iWo0sxaoaKfh5fosnwb6Al9solrMrDVpQS8s\nE0BEvNZEtZhZK9MiXoULbC7pnPoWRsTVNajHzFqJlnQa3hboAAUaCGVmBSLatpCW5ayI+HGTVWJm\nrUrydse8q8iuYp+lmVlNNJPxk1mVC8uBTVaFmbVKLeICT0RkfUuamVmDtaTTcDOzmmoRLUszs1or\nUFY6LM0sH6L2b3esJoelmeVDNIsHZGTlsDSz3BQnKh2WZpYTQYu5g8fMrKYKlJUOSzPLS/N4qG9W\nDkszy4WvhpuZZeSWpZlZBsWJSoelmeXF4yzNzCpzn6WZWUZuWZqZZdBSHv5rZlYzyWl4cdLSYWlm\nuSnQWXih+lfNrEVRg/6v4t6k30uaK2lcybxNJT0s6dX0z01Klp0vabKkVyQdXGn/Dkszy42Ufcrg\nFuCQ1eYNBUZFRE9gVPodSbsCxwC7pdv8WlLbcjt3WJpZLlb1WWadKomIR4HV3x02CLg1/XwrMLhk\n/l0RsTQipgCTgf7l9u+wNLN8NKBVuRZ9m50jYlb6eTbQOf28NTC9ZL0Z6bx6+QKPmeWmgSHYSdKY\nku83RsSNWTeOiJAUDTpiCYelmeUmy4WbEvMjYo8GHmKOpC4RMUtSF2BuOn8m0LVkvW3SefXyaXgN\nzZg+nUMP+jz9dt+NPXr34vpf/uJjy6+75io6tGvD/Pnzc6rQKtmm88b8/cbv8tyfL+TZP13IGccO\nAOBTO27Nv249l2dGXMCfrj2Fjhus97Htum65CfMeu4qzjx+YQ9XFIJJB6VmnRhoJnJB+PgG4t2T+\nMZLaSdoW6Ak8XW5HblnWUF1dHZdefiW9+/Rl4cKF7Lf3Hnz+gAPZZZddmTF9OqP+8TBdu3XLu0wr\nY/mKlQy9+i+MnTiDDuu34/E7f8iopybym/86jqHX3MPoZyfz9UF7870TBvLjX9//4XaXn3skDz02\nPsfKi6Ga7w2XNBwYQHK6PgO4GLgMGCHpRGAacBRARIyXNAKYACwHzoiIFWVrrVql9glbdulC7z59\nAejYsSM77bwLs2YmLf0fnncOP7308kLdG9sazZ7/LmMnzgBg0XtLmThlNlttvjE7dNuC0c9OBuCR\nJycyeGDvD7c5YsCnmTpzARNem51LzUVSzXGWEXFsRHSJiHUiYpuIuDkiFkTEwIjoGREHRMSbJesP\ni4jtI2KniPhbpf07LJvItKlTeeGF59mj/17cN/JettpqKz716d3zLssaoFuXTem90zY8M24qL78+\niyMGfBqAIw/syzadk7HOG7Rfl3O/eSDDfvtAnqUWQhOdhldNTcNS0iHp6PjJkobW8ljN2aJFixhy\nzFe4/MprqKur48qfX8pFF/8477KsATZovy7Dr/w25135ZxYuXsIpl9zByUftx2N3/IAO67fjg2XJ\nGdxFp36BX97+CIvf/yDniougunfw1FrN+izT0fDXAweSjGF6RtLIiJhQq2M2R8uWLWPI0V/h6GOO\nY9DgIxk37iWmTp3CZ/ZMTttmzpjBvnv3439HP0XnLbfMuVpbk7q6Ngy/8iTu/tsY7n3kBQAmTZ3D\nEadfD8AO3bbg0P12A2DPXt350gG9GXb2YDbq2J6VK4MlHyzjhrsfza3+Zmvtxk82uVpe4OkPTI6I\n1wEk3UUyar7VhGVEcPop32annXfmzLPPAaBXr08xdcacD9fZdcdtefTxZ+jUqVNeZVoFN1w8hFem\nzOa62x/5cN7mm3Rg3luLkMTQkw7mpj+NBuCAE6/9cJ0LTzmMxe8tdVCWUaCsrGlYrmmE/F6rryTp\nZOBkoMVdGX7i8ccYfsdt7NbrU3xmzz4AXPLjYRx86GE5V2ZZ7dN7O4YcvhcvTZrJk3clPUkX/2ok\nO3TdglOO3h+Aex8Zyx/vfTLPMgsp6bMsTlzmPnQoHYF/I0Dffns0enR9c7TPZ/dl0dKVZdeZMGlK\nE1VjjfH42Ndp3+c7n5j/IBO4fvi/ym7rizyVFScqaxuWDR4hb2atTIHSspZh+QzQMx0dP5PkcUjH\n1fB4ZlYwPg0HImK5pO8ADwJtgd9HhG9pMLMPFScqa9xnGREPAO64MbM1K1Ba5n6Bx8xaJ9Hgpw7l\nymFpZvnwoHQzs2wKlJUOSzPLUYHS0mFpZjlpHg/IyMphaWa5cZ+lmVkFolBn4Q5LM8tPkd4U4LA0\ns9wUKCsdlmaWnwJlpcPSzHJSsE5Lh6WZ5cZDh8zMKhDuszQzy6RAWemwNLMcFSgtHZZmlhv3WZqZ\nZdCmOFnpsDSzHDkszczK85PSzcyy8JPSzcyyKVBWOizNLEcFSkuHpZnlxE9KNzPLxH2WZmYVVPuh\nQ5KmAguBFcDyiNhD0qbA3UAPYCpwVES81Zj9t6lOmWZmjaAGTNl8LiJ6R8Qe6fehwKiI6AmMSr83\nisPSzHLTRso8NdIg4Nb0863A4EbX2tgNzczWVgMblp0kjSmZTl5tdwH8Q9KzJcs6R8Ss9PNsoHNj\na3WfpZnlo+GD0ueXnF6vyb4RMVPSFsDDkiaWLoyIkBSNqBRwy9LMclW9TsuImJn+ORe4B+gPzJHU\nBSD9c25jK3VYmlkuVj0pPetUdl/SBpI6rvoMHASMA0YCJ6SrnQDc29h6fRpuZrmp4tChzsA96XvI\n64A7I+Lvkp4BRkg6EZgGHNXYAzgszSw31RqUHhGvA7uvYf4CYGA1juGwNLPc+HZHM7MsipOVDksz\ny0+BstJhaWb5kFibO3OanMPSzPJTnKx0WJpZfgqUlQ5LM8tPgc7CHZZmlhc/Kd3MrKJVtzsWhe8N\nNzPLwC1LM8tNkVqWDkszy437LM3MKkgGpeddRXYOSzPLj8PSzKwyn4abmWXgCzxmZhkUKCsdlmaW\nowKlpcPSzHJTpD5LRTT6NbpVJ2keyUuFWrpOwPy8i7CqaC3/lt0jYvNq7lDS30n+/rKaHxGHVLOG\nhmhWYdlaSBpT4WXxVhD+t2w9fG+4mVkGDkszswwclvm4Me8CrGr8b9lKuM/SzCwDtyzNzDJwWJqZ\nZeCwNDPLwGHZhCS1zbsGW3uSdpL0GUnr+N+09fAFniYgaceImJR+bhsRK/KuyRpH0pHAz4CZ6TQG\nuCUi3s21MKs5tyxrTNLhwFhJdwJExAq3RopJ0jrA0cCJETEQuBfoCvxQ0oa5Fmc157CsIUkbAN8B\nzgY+kHQ7ODALbkOgZ/r5HuA+YB3gOKlIT2e0hnJY1lBELAa+BdwJfB9YrzQw86zNGi4ilgFXA0dK\n2i8iVgKjgbHAvrkWZzXnsKyxiHgjIhZFxHzgFKD9qsCU1FfSzvlWaA30b+Ah4HhJ+0fEioi4E9gK\n2D3f0qyW/DzLJhQRCySdAlwhaSLQFvhczmVZA0TEEkl3AAGcn/7HbinQGZiVa3FWUw7LJhYR8yW9\nCBwKHBgRM/KuyRomIt6SdBMwgeRsYQnwtYiYk29lVkseOtTEJG0CjADOjYgX867H1k56oS7S/ktr\nwRyWOZC0XkQsybsOM8vOYWlmloGvhpuZZeCwNDPLwGFpZpaBw9LMLAOHZQshaYWksZLGSfpvSeuv\nxb4GSLov/fxFSUPLrLuxpNMbcYxLJH0/6/zV1rlF0lcacKweksY1tEazUg7LluP9iOgdEb2AD4BT\nSxcq0eB/74gYGRGXlVllY6DBYWlWNA7LlunfwA5pi+oVSX8ExgFdJR0k6QlJz6Ut0A4Akg6RNFHS\nc8CRq3Yk6RuSfpV+7izpHkkvpNM+wGXA9mmr9op0vfMkPSPpRUk/KtnXhZImSRoN7FTpR0g6Kd3P\nC5L+vFpr+QBJY9L9HZ6u31bSFSXHPmVt/yLNVnFYtjCS6khupXwpndUT+HVE7AYsBi4CDoiIviQP\nrj1H0nrATcARQD9gy3p2fx3wvxGxO9AXGA8MBV5LW7XnSTooPWZ/oDfQT9L+kvoBx6TzDgP2zPBz\n/hIRe6bHexk4sWRZj/QYXwBuSH/DicA7EbFnuv+TJG2b4ThmFfne8JajvaSx6ed/AzeTPAlnWkQ8\nmc7fG9gVeCx99OK6wBPAzsCUiHgVIH0q0slrOMbnga/Dh4+Yeye9fbPUQen0fPq9A0l4dgTuiYj3\n0mOMzPCbekn6KcmpfgfgwZJlI9JbDF+V9Hr6Gw4CPl3Sn7lReuxJGY5lVpbDsuV4PyJ6l85IA3Fx\n6Szg4Yg4drX1PrbdWhJwaUT8drVjnN2Ifd0CDI6IFyR9AxhQsmz1W88iPfaZEVEaqkjq0Yhjm32M\nT8NblyeBz0raAZInuUvaEZgI9JC0fbresfVsPwo4Ld22raSNgIUkrcZVHgS+VdIXurWkLYBHgcGS\n2kvqSHLKX0lHYFb6Oochqy37qqQ2ac3bAa+kxz4tXR9JO6ZPqzdba25ZtiIRMS9toQ2X1C6dfVFE\nTJJ0MnC/pPdITuM7rmEXZwE3SjoRWAGcFhFPSHosHZrzt7TfchfgibRlu4jk8WXPSbobeAGYCzyT\noeT/BJ4C5qV/ltb0f8DTJK95ODV9zuTvSPoyn1Ny8HnA4Gx/O2bl+UEaZmYZ+DTczCwDh6WZWQYO\nSzOzDByWZmYZOCzNzDJwWJqZZeCwNDPL4P8B8E1QzdH4NkUAAAAASUVORK5CYII=\n",
   "text/plain": "<matplotlib.figure.Figure at 0x482e710>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

```{.python .input  n=58}
print(accuracy_score(test["trend"], test["predict"]))
print(confusion_matrix(test["trend"], test["predict"]))
print(classification_report(test["trend"], test["predict"]))
```

```{.json .output n=58}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0.526845637584\n[[ 20 258]\n [ 24 294]]\n             precision    recall  f1-score   support\n\n          0       0.45      0.07      0.12       278\n          1       0.53      0.92      0.68       318\n\navg / total       0.50      0.53      0.42       596\n\n"
 }
]
```

## Feature Importance
As expected, importance of features - 'moving sum
sentiment scores' is higher than daywise sentiment scores!

```{.python .input  n=59}
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

```{.json .output n=59}
[
 {
  "data": {
   "image/png": "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\nmmmlOhnXWVU/TbIfzfRib0myZVW9bsw1rk/yTJp5YE9LcgrNVFZ3o5k+7HSa+XhvmPtZJPWZe1Yl\n9V5VXU1zuPxI4JfA44GjaPYUXt1+/8WB7T9DM5H9d4CDgGcAv2+3P32OMq+mmZLpdsDLgH+hmZx/\n5jmPp5mT9Cc0U2s9geYM9H1ZNW5yfV/XN4C7Av9Bc4b7M2iC+V40Fz94CquGIIysnbLqEcBxNHO4\nvgC4N034PoBVJy6NXVX9AnggzXCOf0/yTx3U+ALN7/bLNGNzn0+zt/t+rPoAM9LvSNL0pZm6TpKk\njU+Sc2nC/zZVtdA91JKmwD2rkqQNWpKbtCe6DS9/Js3wj88aVKUNl3tWJUkbtCR3prn4wpk088hu\nTrM39d40w0b2qarvTq+FkhbCsCpJ2qAl+Uuacb/3p5lX9cY0Y4vPBF5TVd+bYvMkLZBhVZIkSb3l\nmFVJkiT11iY1z+oOO+xQu+2227SbIUmStEk7//zzf15Vi+ez7SYVVnfbbTeWLVs27WZIkiRt0pL8\nYN1bNRwGIEmSpN4yrEqSJKm3DKuSJEnqLcOqJEmSesuwKkmSpN4yrEqSJKm3DKuSJEnqLcOqJEmS\nesuwKkmSpN4yrEqSJKm3DKuSJEnqLcOqJEmSesuwKkmSpN4yrEqSJKm3DKuSJEnqLcOqJEmSesuw\nKkmSpN4yrEqSJKm3DKuSJEnqLcOqJEmSesuwKkmSpN7abNoN2NAk3T13VXfPLUmStCEyrG4AugzI\nYEiWJEn95TAASZIk9ZZhVZIkSb1lWJUkSVJvGVYlSZLUW4ZVSZIk9ZZhVZIkSb1lWJUkSVJvGVYl\nSZLUW4ZVSZIk9ZZhVZIkSb1lWJUkSVJvGVYlSZLUW4ZVSZIk9dZm026A+ivp7rmruntuSZK08XDP\nqiRJknrLsCpJkqTeMqxKkiSptwyrkiRJ6i3DqiRJknrLsCpJkqTeMqxKkiSptwyrkiRJ6i3DqiRJ\nknrLsCpJkqTeMqxKkiSptwyrkiRJ6i3DqiRJknprs2k3QBqUdPfcVd09tyRJ6oZ7ViVJktRbhlVJ\nkiT1lmFVkiRJvWVYlSRJUm8ZViVJktRbhlVJkiT1lmFVkiRJvWVYlSRJUm8ZViVJktRbhlVJkiT1\nlmFVkiRJvWVYlSRJUm9NNawm2T/JZUmWJzl6lvV3SHJOkmuTvHhg+a5JzkpySZKLkxw52ZZLkiRp\n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\nQldzAAAAAElFTkSuQmCC\n",
   "text/plain": "<matplotlib.figure.Figure at 0x16197cc0>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

### Gradient Boosting Classifier

```{.python .input  n=61}
clf = GradientBoostingClassifier(n_estimators=1000, max_depth=10)
#clf = AdaBoostClassifier(n_estimators=100)
#clf = linear_model.LinearRegression()

clf.fit(X,y)
test["predict"] = clf.predict(test_X)
test.head(5)
```

```{.json .output n=61}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>PosCount</th>\n      <th>NegCount</th>\n      <th>TrustCount</th>\n      <th>AngerCount</th>\n      <th>AnticipationCount</th>\n      <th>DisgustCount</th>\n      <th>FearCount</th>\n      <th>JoyCount</th>\n      <th>SadnessCount</th>\n      <th>SurpriseCount</th>\n      <th>...</th>\n      <th>TrustCount_cv</th>\n      <th>AngerCount_cv</th>\n      <th>AnticipationCount_cv</th>\n      <th>DisgustCount_cv</th>\n      <th>FearCount_cv</th>\n      <th>JoyCount_cv</th>\n      <th>SadnessCount_cv</th>\n      <th>SurpriseCount_cv</th>\n      <th>trend</th>\n      <th>predict</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1393</th>\n      <td>30</td>\n      <td>35</td>\n      <td>27</td>\n      <td>20</td>\n      <td>9</td>\n      <td>10</td>\n      <td>31</td>\n      <td>6</td>\n      <td>17</td>\n      <td>9</td>\n      <td>...</td>\n      <td>99.0</td>\n      <td>82.0</td>\n      <td>46.0</td>\n      <td>44.0</td>\n      <td>125.0</td>\n      <td>27.0</td>\n      <td>70.0</td>\n      <td>29.0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1394</th>\n      <td>25</td>\n      <td>33</td>\n      <td>20</td>\n      <td>20</td>\n      <td>12</td>\n      <td>5</td>\n      <td>29</td>\n      <td>10</td>\n      <td>14</td>\n      <td>3</td>\n      <td>...</td>\n      <td>107.0</td>\n      <td>77.0</td>\n      <td>54.0</td>\n      <td>38.0</td>\n      <td>112.0</td>\n      <td>41.0</td>\n      <td>63.0</td>\n      <td>30.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1395</th>\n      <td>23</td>\n      <td>25</td>\n      <td>23</td>\n      <td>13</td>\n      <td>12</td>\n      <td>5</td>\n      <td>11</td>\n      <td>10</td>\n      <td>10</td>\n      <td>7</td>\n      <td>...</td>\n      <td>109.0</td>\n      <td>76.0</td>\n      <td>57.0</td>\n      <td>36.0</td>\n      <td>98.0</td>\n      <td>42.0</td>\n      <td>59.0</td>\n      <td>30.0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1396</th>\n      <td>23</td>\n      <td>22</td>\n      <td>10</td>\n      <td>9</td>\n      <td>5</td>\n      <td>6</td>\n      <td>15</td>\n      <td>4</td>\n      <td>11</td>\n      <td>2</td>\n      <td>...</td>\n      <td>92.0</td>\n      <td>65.0</td>\n      <td>53.0</td>\n      <td>32.0</td>\n      <td>82.0</td>\n      <td>40.0</td>\n      <td>53.0</td>\n      <td>23.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1397</th>\n      <td>27</td>\n      <td>22</td>\n      <td>19</td>\n      <td>13</td>\n      <td>11</td>\n      <td>3</td>\n      <td>18</td>\n      <td>8</td>\n      <td>9</td>\n      <td>4</td>\n      <td>...</td>\n      <td>85.0</td>\n      <td>68.0</td>\n      <td>47.0</td>\n      <td>27.0</td>\n      <td>88.0</td>\n      <td>36.0</td>\n      <td>52.0</td>\n      <td>19.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 22 columns</p>\n</div>",
   "text/plain": "      PosCount  NegCount  TrustCount  AngerCount  AnticipationCount  \\\n1393        30        35          27          20                  9   \n1394        25        33          20          20                 12   \n1395        23        25          23          13                 12   \n1396        23        22          10           9                  5   \n1397        27        22          19          13                 11   \n\n      DisgustCount  FearCount  JoyCount  SadnessCount  SurpriseCount   ...     \\\n1393            10         31         6            17              9   ...      \n1394             5         29        10            14              3   ...      \n1395             5         11        10            10              7   ...      \n1396             6         15         4            11              2   ...      \n1397             3         18         8             9              4   ...      \n\n      TrustCount_cv  AngerCount_cv  AnticipationCount_cv  DisgustCount_cv  \\\n1393           99.0           82.0                  46.0             44.0   \n1394          107.0           77.0                  54.0             38.0   \n1395          109.0           76.0                  57.0             36.0   \n1396           92.0           65.0                  53.0             32.0   \n1397           85.0           68.0                  47.0             27.0   \n\n      FearCount_cv  JoyCount_cv  SadnessCount_cv  SurpriseCount_cv  trend  \\\n1393         125.0         27.0             70.0              29.0      0   \n1394         112.0         41.0             63.0              30.0      1   \n1395          98.0         42.0             59.0              30.0      0   \n1396          82.0         40.0             53.0              23.0      1   \n1397          88.0         36.0             52.0              19.0      1   \n\n      predict  \n1393        0  \n1394        1  \n1395        0  \n1396        1  \n1397        1  \n\n[5 rows x 22 columns]"
  },
  "execution_count": 61,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=62}
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

```{.json .output n=62}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Confusion matrix\n[[116 162]\n [113 205]]\n"
 },
 {
  "data": {
   "image/png": "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\nAABJRU5ErkJggg==\n",
   "text/plain": "<matplotlib.figure.Figure at 0x25eb5c0>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0.538590604027\n[[116 162]\n [113 205]]\n             precision    recall  f1-score   support\n\n          0       0.51      0.42      0.46       278\n          1       0.56      0.64      0.60       318\n\navg / total       0.53      0.54      0.53       596\n\n"
 }
]
```

```{.python .input  n=63}
print(accuracy_score(test["trend"], test["predict"]))
print(confusion_matrix(test["trend"], test["predict"]))
print(classification_report(test["trend"], test["predict"]))
```

```{.json .output n=63}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0.538590604027\n[[116 162]\n [113 205]]\n             precision    recall  f1-score   support\n\n          0       0.51      0.42      0.46       278\n          1       0.56      0.64      0.60       318\n\navg / total       0.53      0.54      0.53       596\n\n"
 }
]
```

### AdaBoost Classifier

```{.python .input  n=64}
#clf = GradientBoostingClassifier(n_estimators=1000, max_depth=10)
clf = AdaBoostClassifier(n_estimators=100)
#clf = linear_model.LinearRegression()

clf.fit(X,y)
test["predict"] = clf.predict(test_X)
test.head(5)
```

```{.json .output n=64}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>PosCount</th>\n      <th>NegCount</th>\n      <th>TrustCount</th>\n      <th>AngerCount</th>\n      <th>AnticipationCount</th>\n      <th>DisgustCount</th>\n      <th>FearCount</th>\n      <th>JoyCount</th>\n      <th>SadnessCount</th>\n      <th>SurpriseCount</th>\n      <th>...</th>\n      <th>TrustCount_cv</th>\n      <th>AngerCount_cv</th>\n      <th>AnticipationCount_cv</th>\n      <th>DisgustCount_cv</th>\n      <th>FearCount_cv</th>\n      <th>JoyCount_cv</th>\n      <th>SadnessCount_cv</th>\n      <th>SurpriseCount_cv</th>\n      <th>trend</th>\n      <th>predict</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1393</th>\n      <td>30</td>\n      <td>35</td>\n      <td>27</td>\n      <td>20</td>\n      <td>9</td>\n      <td>10</td>\n      <td>31</td>\n      <td>6</td>\n      <td>17</td>\n      <td>9</td>\n      <td>...</td>\n      <td>99.0</td>\n      <td>82.0</td>\n      <td>46.0</td>\n      <td>44.0</td>\n      <td>125.0</td>\n      <td>27.0</td>\n      <td>70.0</td>\n      <td>29.0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1394</th>\n      <td>25</td>\n      <td>33</td>\n      <td>20</td>\n      <td>20</td>\n      <td>12</td>\n      <td>5</td>\n      <td>29</td>\n      <td>10</td>\n      <td>14</td>\n      <td>3</td>\n      <td>...</td>\n      <td>107.0</td>\n      <td>77.0</td>\n      <td>54.0</td>\n      <td>38.0</td>\n      <td>112.0</td>\n      <td>41.0</td>\n      <td>63.0</td>\n      <td>30.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1395</th>\n      <td>23</td>\n      <td>25</td>\n      <td>23</td>\n      <td>13</td>\n      <td>12</td>\n      <td>5</td>\n      <td>11</td>\n      <td>10</td>\n      <td>10</td>\n      <td>7</td>\n      <td>...</td>\n      <td>109.0</td>\n      <td>76.0</td>\n      <td>57.0</td>\n      <td>36.0</td>\n      <td>98.0</td>\n      <td>42.0</td>\n      <td>59.0</td>\n      <td>30.0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1396</th>\n      <td>23</td>\n      <td>22</td>\n      <td>10</td>\n      <td>9</td>\n      <td>5</td>\n      <td>6</td>\n      <td>15</td>\n      <td>4</td>\n      <td>11</td>\n      <td>2</td>\n      <td>...</td>\n      <td>92.0</td>\n      <td>65.0</td>\n      <td>53.0</td>\n      <td>32.0</td>\n      <td>82.0</td>\n      <td>40.0</td>\n      <td>53.0</td>\n      <td>23.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1397</th>\n      <td>27</td>\n      <td>22</td>\n      <td>19</td>\n      <td>13</td>\n      <td>11</td>\n      <td>3</td>\n      <td>18</td>\n      <td>8</td>\n      <td>9</td>\n      <td>4</td>\n      <td>...</td>\n      <td>85.0</td>\n      <td>68.0</td>\n      <td>47.0</td>\n      <td>27.0</td>\n      <td>88.0</td>\n      <td>36.0</td>\n      <td>52.0</td>\n      <td>19.0</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 22 columns</p>\n</div>",
   "text/plain": "      PosCount  NegCount  TrustCount  AngerCount  AnticipationCount  \\\n1393        30        35          27          20                  9   \n1394        25        33          20          20                 12   \n1395        23        25          23          13                 12   \n1396        23        22          10           9                  5   \n1397        27        22          19          13                 11   \n\n      DisgustCount  FearCount  JoyCount  SadnessCount  SurpriseCount   ...     \\\n1393            10         31         6            17              9   ...      \n1394             5         29        10            14              3   ...      \n1395             5         11        10            10              7   ...      \n1396             6         15         4            11              2   ...      \n1397             3         18         8             9              4   ...      \n\n      TrustCount_cv  AngerCount_cv  AnticipationCount_cv  DisgustCount_cv  \\\n1393           99.0           82.0                  46.0             44.0   \n1394          107.0           77.0                  54.0             38.0   \n1395          109.0           76.0                  57.0             36.0   \n1396           92.0           65.0                  53.0             32.0   \n1397           85.0           68.0                  47.0             27.0   \n\n      FearCount_cv  JoyCount_cv  SadnessCount_cv  SurpriseCount_cv  trend  \\\n1393         125.0         27.0             70.0              29.0      0   \n1394         112.0         41.0             63.0              30.0      1   \n1395          98.0         42.0             59.0              30.0      0   \n1396          82.0         40.0             53.0              23.0      1   \n1397          88.0         36.0             52.0              19.0      1   \n\n      predict  \n1393        0  \n1394        1  \n1395        0  \n1396        1  \n1397        0  \n\n[5 rows x 22 columns]"
  },
  "execution_count": 64,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=65}
print(accuracy_score(test["trend"], test["predict"]))
print(confusion_matrix(test["trend"], test["predict"]))
print(classification_report(test["trend"], test["predict"]))
```

```{.json .output n=65}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0.468120805369\n[[ 97 181]\n [136 182]]\n             precision    recall  f1-score   support\n\n          0       0.42      0.35      0.38       278\n          1       0.50      0.57      0.53       318\n\navg / total       0.46      0.47      0.46       596\n\n"
 }
]
```

### Neural Network

```{.python .input  n=82}
#clf = GradientBoostingClassifier(n_estimators=1000, max_depth=10)
#clf = AdaBoostClassifier(n_estimators=100)
#clf = linear_model.LinearRegression()
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,10), random_state=1)

clf.fit(X,y)
test["predict"] = clf.predict(test_X)
test.head(5)
```

```{.json .output n=82}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>PosCount</th>\n      <th>NegCount</th>\n      <th>TrustCount</th>\n      <th>AngerCount</th>\n      <th>AnticipationCount</th>\n      <th>DisgustCount</th>\n      <th>FearCount</th>\n      <th>JoyCount</th>\n      <th>SadnessCount</th>\n      <th>SurpriseCount</th>\n      <th>...</th>\n      <th>TrustCount_cv</th>\n      <th>AngerCount_cv</th>\n      <th>AnticipationCount_cv</th>\n      <th>DisgustCount_cv</th>\n      <th>FearCount_cv</th>\n      <th>JoyCount_cv</th>\n      <th>SadnessCount_cv</th>\n      <th>SurpriseCount_cv</th>\n      <th>trend</th>\n      <th>predict</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1393</th>\n      <td>30</td>\n      <td>35</td>\n      <td>27</td>\n      <td>20</td>\n      <td>9</td>\n      <td>10</td>\n      <td>31</td>\n      <td>6</td>\n      <td>17</td>\n      <td>9</td>\n      <td>...</td>\n      <td>99.0</td>\n      <td>82.0</td>\n      <td>46.0</td>\n      <td>44.0</td>\n      <td>125.0</td>\n      <td>27.0</td>\n      <td>70.0</td>\n      <td>29.0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1394</th>\n      <td>25</td>\n      <td>33</td>\n      <td>20</td>\n      <td>20</td>\n      <td>12</td>\n      <td>5</td>\n      <td>29</td>\n      <td>10</td>\n      <td>14</td>\n      <td>3</td>\n      <td>...</td>\n      <td>107.0</td>\n      <td>77.0</td>\n      <td>54.0</td>\n      <td>38.0</td>\n      <td>112.0</td>\n      <td>41.0</td>\n      <td>63.0</td>\n      <td>30.0</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1395</th>\n      <td>23</td>\n      <td>25</td>\n      <td>23</td>\n      <td>13</td>\n      <td>12</td>\n      <td>5</td>\n      <td>11</td>\n      <td>10</td>\n      <td>10</td>\n      <td>7</td>\n      <td>...</td>\n      <td>109.0</td>\n      <td>76.0</td>\n      <td>57.0</td>\n      <td>36.0</td>\n      <td>98.0</td>\n      <td>42.0</td>\n      <td>59.0</td>\n      <td>30.0</td>\n      <td>0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1396</th>\n      <td>23</td>\n      <td>22</td>\n      <td>10</td>\n      <td>9</td>\n      <td>5</td>\n      <td>6</td>\n      <td>15</td>\n      <td>4</td>\n      <td>11</td>\n      <td>2</td>\n      <td>...</td>\n      <td>92.0</td>\n      <td>65.0</td>\n      <td>53.0</td>\n      <td>32.0</td>\n      <td>82.0</td>\n      <td>40.0</td>\n      <td>53.0</td>\n      <td>23.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1397</th>\n      <td>27</td>\n      <td>22</td>\n      <td>19</td>\n      <td>13</td>\n      <td>11</td>\n      <td>3</td>\n      <td>18</td>\n      <td>8</td>\n      <td>9</td>\n      <td>4</td>\n      <td>...</td>\n      <td>85.0</td>\n      <td>68.0</td>\n      <td>47.0</td>\n      <td>27.0</td>\n      <td>88.0</td>\n      <td>36.0</td>\n      <td>52.0</td>\n      <td>19.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 22 columns</p>\n</div>",
   "text/plain": "      PosCount  NegCount  TrustCount  AngerCount  AnticipationCount  \\\n1393        30        35          27          20                  9   \n1394        25        33          20          20                 12   \n1395        23        25          23          13                 12   \n1396        23        22          10           9                  5   \n1397        27        22          19          13                 11   \n\n      DisgustCount  FearCount  JoyCount  SadnessCount  SurpriseCount   ...     \\\n1393            10         31         6            17              9   ...      \n1394             5         29        10            14              3   ...      \n1395             5         11        10            10              7   ...      \n1396             6         15         4            11              2   ...      \n1397             3         18         8             9              4   ...      \n\n      TrustCount_cv  AngerCount_cv  AnticipationCount_cv  DisgustCount_cv  \\\n1393           99.0           82.0                  46.0             44.0   \n1394          107.0           77.0                  54.0             38.0   \n1395          109.0           76.0                  57.0             36.0   \n1396           92.0           65.0                  53.0             32.0   \n1397           85.0           68.0                  47.0             27.0   \n\n      FearCount_cv  JoyCount_cv  SadnessCount_cv  SurpriseCount_cv  trend  \\\n1393         125.0         27.0             70.0              29.0      0   \n1394         112.0         41.0             63.0              30.0      1   \n1395          98.0         42.0             59.0              30.0      0   \n1396          82.0         40.0             53.0              23.0      1   \n1397          88.0         36.0             52.0              19.0      1   \n\n      predict  \n1393        0  \n1394        0  \n1395        1  \n1396        1  \n1397        1  \n\n[5 rows x 22 columns]"
  },
  "execution_count": 82,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=83}
print(accuracy_score(test["trend"], test["predict"]))
print(confusion_matrix(test["trend"], test["predict"]))
print(classification_report(test["trend"], test["predict"]))
```

```{.json .output n=83}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0.541946308725\n[[ 38 240]\n [ 33 285]]\n             precision    recall  f1-score   support\n\n          0       0.54      0.14      0.22       278\n          1       0.54      0.90      0.68       318\n\navg / total       0.54      0.54      0.46       596\n\n"
 }
]
```

### Naive Bayes

```{.python .input  n=70}
# Naive Bayes Algo
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, y)
test["predict"] = clf.predict(test_X)
test.head(5)
```

```{.json .output n=70}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>PosCount</th>\n      <th>NegCount</th>\n      <th>TrustCount</th>\n      <th>AngerCount</th>\n      <th>AnticipationCount</th>\n      <th>DisgustCount</th>\n      <th>FearCount</th>\n      <th>JoyCount</th>\n      <th>SadnessCount</th>\n      <th>SurpriseCount</th>\n      <th>...</th>\n      <th>TrustCount_cv</th>\n      <th>AngerCount_cv</th>\n      <th>AnticipationCount_cv</th>\n      <th>DisgustCount_cv</th>\n      <th>FearCount_cv</th>\n      <th>JoyCount_cv</th>\n      <th>SadnessCount_cv</th>\n      <th>SurpriseCount_cv</th>\n      <th>trend</th>\n      <th>predict</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1393</th>\n      <td>30</td>\n      <td>35</td>\n      <td>27</td>\n      <td>20</td>\n      <td>9</td>\n      <td>10</td>\n      <td>31</td>\n      <td>6</td>\n      <td>17</td>\n      <td>9</td>\n      <td>...</td>\n      <td>99.0</td>\n      <td>82.0</td>\n      <td>46.0</td>\n      <td>44.0</td>\n      <td>125.0</td>\n      <td>27.0</td>\n      <td>70.0</td>\n      <td>29.0</td>\n      <td>0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1394</th>\n      <td>25</td>\n      <td>33</td>\n      <td>20</td>\n      <td>20</td>\n      <td>12</td>\n      <td>5</td>\n      <td>29</td>\n      <td>10</td>\n      <td>14</td>\n      <td>3</td>\n      <td>...</td>\n      <td>107.0</td>\n      <td>77.0</td>\n      <td>54.0</td>\n      <td>38.0</td>\n      <td>112.0</td>\n      <td>41.0</td>\n      <td>63.0</td>\n      <td>30.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1395</th>\n      <td>23</td>\n      <td>25</td>\n      <td>23</td>\n      <td>13</td>\n      <td>12</td>\n      <td>5</td>\n      <td>11</td>\n      <td>10</td>\n      <td>10</td>\n      <td>7</td>\n      <td>...</td>\n      <td>109.0</td>\n      <td>76.0</td>\n      <td>57.0</td>\n      <td>36.0</td>\n      <td>98.0</td>\n      <td>42.0</td>\n      <td>59.0</td>\n      <td>30.0</td>\n      <td>0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1396</th>\n      <td>23</td>\n      <td>22</td>\n      <td>10</td>\n      <td>9</td>\n      <td>5</td>\n      <td>6</td>\n      <td>15</td>\n      <td>4</td>\n      <td>11</td>\n      <td>2</td>\n      <td>...</td>\n      <td>92.0</td>\n      <td>65.0</td>\n      <td>53.0</td>\n      <td>32.0</td>\n      <td>82.0</td>\n      <td>40.0</td>\n      <td>53.0</td>\n      <td>23.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1397</th>\n      <td>27</td>\n      <td>22</td>\n      <td>19</td>\n      <td>13</td>\n      <td>11</td>\n      <td>3</td>\n      <td>18</td>\n      <td>8</td>\n      <td>9</td>\n      <td>4</td>\n      <td>...</td>\n      <td>85.0</td>\n      <td>68.0</td>\n      <td>47.0</td>\n      <td>27.0</td>\n      <td>88.0</td>\n      <td>36.0</td>\n      <td>52.0</td>\n      <td>19.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 22 columns</p>\n</div>",
   "text/plain": "      PosCount  NegCount  TrustCount  AngerCount  AnticipationCount  \\\n1393        30        35          27          20                  9   \n1394        25        33          20          20                 12   \n1395        23        25          23          13                 12   \n1396        23        22          10           9                  5   \n1397        27        22          19          13                 11   \n\n      DisgustCount  FearCount  JoyCount  SadnessCount  SurpriseCount   ...     \\\n1393            10         31         6            17              9   ...      \n1394             5         29        10            14              3   ...      \n1395             5         11        10            10              7   ...      \n1396             6         15         4            11              2   ...      \n1397             3         18         8             9              4   ...      \n\n      TrustCount_cv  AngerCount_cv  AnticipationCount_cv  DisgustCount_cv  \\\n1393           99.0           82.0                  46.0             44.0   \n1394          107.0           77.0                  54.0             38.0   \n1395          109.0           76.0                  57.0             36.0   \n1396           92.0           65.0                  53.0             32.0   \n1397           85.0           68.0                  47.0             27.0   \n\n      FearCount_cv  JoyCount_cv  SadnessCount_cv  SurpriseCount_cv  trend  \\\n1393         125.0         27.0             70.0              29.0      0   \n1394         112.0         41.0             63.0              30.0      1   \n1395          98.0         42.0             59.0              30.0      0   \n1396          82.0         40.0             53.0              23.0      1   \n1397          88.0         36.0             52.0              19.0      1   \n\n      predict  \n1393        1  \n1394        1  \n1395        1  \n1396        1  \n1397        1  \n\n[5 rows x 22 columns]"
  },
  "execution_count": 70,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=71}
print(accuracy_score(test["trend"], test["predict"]))
print(confusion_matrix(test["trend"], test["predict"]))
print(classification_report(test["trend"], test["predict"]))
```

```{.json .output n=71}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0.506711409396\n[[ 78 200]\n [ 94 224]]\n             precision    recall  f1-score   support\n\n          0       0.45      0.28      0.35       278\n          1       0.53      0.70      0.60       318\n\navg / total       0.49      0.51      0.48       596\n\n"
 }
]
```

### Conclusion:
The maximum accuracy of predicting overall stock market trend is
55%. It implies that sentiment of news headlines have the impact on stock market
trend.

```{.python .input}

```
