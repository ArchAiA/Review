
# TASK 1: read data into a DataFrame
import pandas as pd
import numpy as np

feat_cols = ['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type']
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', 
                   names=feat_cols, index_col='id')
                   


# TASK 2: briefly explore the data
data.isnull().sum() #Woohoo, no nulls
data.shape
data.describe()
data.head()
data.tail()
data.glass_type.value_counts()



# TASK 3: convert to binary classification problem (1/2/3/4 maps to 0, 5/6/7 maps to 1)
#Can be done the long way: data['binary'] = data.glass_type.map({1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1})
#Or by using numpy
data['binary'] = np.where(data.glass_type < 5, 0, 1)
data.binary.value_counts()



# TASK 4: create a feature matrix (X)
featureList = data.columns[:-2]
#This can also be done the long, more explicit way: featureList = ['ri','na','mg','al','si','k','ca','ba','fe']
X = data[featureList]



# TASK 5: create a response vector (y)
y = data.binary



# TASK 6: split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

# TASK 7: fit a KNN model on the training set using K=5
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# TASK 8: make predictions on the testing set and calculate accuracy
y_pred = knn.predict(X_test)

from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred)



# TASK 9: calculate null accuracy
1 - y.mean() #B/C this is a binary prediction, predicting all zeroes would have resulted in a 76.2% accuracy score (y is the actual observation)



# BONUS: write a for loop that computes test set accuracy for a range of K values
#k_range = (1, 30, 2)
scores2 = [] #***For Last Class Task
scores = []
for k in range(1,99):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append([k, float(metrics.accuracy_score(y_test, y_pred))])

    #***For Last Class Task
    scores2.append(metrics.accuracy_score(y_test, y_pred))

#This finds the highest accuracy score
tempMax = 0
tempNeighbors = 0
for i in scores:
    if i[1] > tempMax:
        tempMax = i[1]
        tempNeighbors = i[0]







'''
HOMEWORK: Glass Identification (aka "Glassification")
'''






# BONUS: plot K versus test set accuracy to choose on optimal value for K
import matplotlib.pyplot as plt
plt.plot(k_range, scores)                       # optimal value is K=1







# TASK 1: read data into a DataFrame
import pandas as pd
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data',
                 names=['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type'],
                 index_col='id')

# TASK 2: briefly explore the data
df.shape
df.head()
df.tail()
df.glass_type.value_counts()
df.isnull().sum()

# TASK 3: convert to binary classification problem (1/2/3/4 maps to 0, 5/6/7 maps to 1)
import numpy as np
df['binary'] = np.where(df.glass_type < 5, 0, 1)                        # method 1
df['binary'] = df.glass_type.map({1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1})   # method 2
df.binary.value_counts()

# TASK 4: create a feature matrix (X)
features = ['ri','na','mg','al','si','k','ca','ba','fe']    # create a list of features
features = df.columns[:-2]      # alternative way: slice 'columns' attribute like a list
X = df[features]                # create DataFrame X by only selecting features

# TASK 5: create a response vector (y)
y = df.binary

# TASK 6: split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

# TASK 7: fit a KNN model on the training set using K=5
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# TASK 8: make predictions on the testing set and calculate accuracy
y_pred = knn.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred)    # 90.7% accuracy


# TASK 9: calculate null accuracy
1 - y.mean()                                    # 76.2% null accuracy

# BONUS: write a for loop that computes test set accuracy for a range of K values
k_range = range(1, 30, 2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

# BONUS: plot K versus test set accuracy to choose on optimal value for K
import matplotlib.pyplot as plt
plt.plot(range(1,99), scores2)