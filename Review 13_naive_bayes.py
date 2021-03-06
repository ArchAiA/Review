



'''
CLASS: Naive Bayes SMS spam classifier
DATA SOURCE: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
'''

## READING IN THE DATA

# read tab-separated file using pandas
import pandas as pd
col_names = ['label', 'msg']
data = pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT5/master/data/SMSSpamCollection.txt',
                   sep='\t', header=None, names=['label', 'msg'])

# examine the data
data.isnull().sum()
data.head()
data.tail()
data.label.value_counts()
data.msg.describe()

# convert label to a binary variable
data['label_bin'] = data.label.map({'ham':0, 'spam':1})

# split into training and testing sets
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(data.msg, data.label_bin, random_state=1)

#CHECKING SPLIT
X_train.shape
X_test.shape
y_train.shape
y_test.shape
#CHECKING SPLIT



## COUNTVECTORIZER: 'convert text into a matrix of token counts'
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklean.feature_extraction.text import CountVectorizer

# start with a simple example
train_simple = ['call you tonight',
                'Call me a cab',
                'please call me... PLEASE!']

# learn the 'VOCABULARY' of the training data
vect = CountVectorizer() #Instantiating the model
vect.fit(train_simple)   #Fitting the model (Fit: Learns the vocabulary)
vect.get_feature_names() #This creates a list of all of the unique (lower case, no punctuation) features, and places them in a list/array
 
# transform training data into a 'DOCUMENT-TERM-MATRIX'
train_simple_dtm = vect.transform(train_simple) #STransforms it into a DOCUMENT TERM MATRIX
train_simple_dtm
train_simple_dtm.toarray()

# examine the vocabulary and document-term matrix together
#ALl this does is print out the dtm array with nice column headers
pd.DataFrame(train_simple_dtm.toarray(), columns=vect.get_feature_names())

# transform testing data into a document-term matrix (using existing vocabulary)
test_simple = ["please don't call me"]
test_simple_dtm = vect.transform(test_simple) #We only run transform on the test data
test_simple_dtm.toarray()
pd.DataFrame(test_simple_dtm.toarray(), columns=vect.get_feature_names())





'''THIS IS THE PROCESS'''
#1) Split data
#2) Fitting: Learning the VOCABULARY
#3) Transforming: Training data into DOCUMENT-TERM-MATRIX
#4) Transofmring: Testing data into DOCUMENT TERM MATRIX

## REPEAT PATTERN WITH SMS DATA

# instantiate the vectorizer
vect = CountVectorizer()
# learn vocabulary and create document-term matrix in a single step
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm
# transform testing data into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm
# store feature names and examine them
X_train_features = vect.get_feature_names() #X_train_features is a list of all of the unique features (no case, no punctuation) in the X_train data
len(X_train_features)
X_train_features[:50]
X_train_features[-50:]
# convert train_dtm to a regular array
X_train_arr = X_train_dtm.toarray()
X_train_arr



## SIMPLE SUMMARIES OF THE TRAINING DATA
# refresher on NumPy
import numpy as np
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) #Create numpy array from list of lists
arr
arr[0, 0]
arr[1, 3]
arr[0, :]
arr[:, 0]
np.sum(arr)
np.sum(arr, axis=0) #Sums the array vertically
np.sum(arr, axis=1) #Sums the array horizontally

# exercise: calculate the number of tokens in the 0th message in train_arr
pd.DataFrame(X_train_arr, columns=vect.get_feature_names())
np.sum(X_train_arr, axis=0) #Get vertical sums of feature appearances
train_arr[0,:].sum()        #Get horizontal sums of feature appearances
np.sum(train_arr[0])        #Get horizontal sums of feature appearances



# exercise: count how many times the 0th token appears across ALL messages in train_arr
X_train_arr[:,0].sum()   #Five times
np.sum(X_train_arr[:,0]) #Five times

# exercise: count how many times EACH token appears across ALL messages in train_arr
np.sum(X_train_arr, axis=0)

# exercise: create a DataFrame of tokens with their counts
#Way 01
sums = list(np.sum(X_train_arr, axis=0)) #Creates a list of the sums of each column
zipped = zip(X_train_features, sums)     #Creates tuples from two equal-length collections
pd.DataFrame(zipped)
data02 = pd.DataFrame(zipped)
#Way 02 - MUCH BETTER as it includes col names via dictionary key
data02 = pd.DataFrame({'token': X_train_features, 'count': np.sum(X_train_arr, axis=0)})





## MODEL BUILDING WITH NAIVE BAYES
## http://scikit-learn.org/stable/modules/naive_bayes.html
# train a Naive Bayes model using train_dtm
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

# make predictions on test data using test_dtm
y_pred = nb.predict(X_test_dtm)
y_pred

# compare predictions to true labels
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred)
print metrics.confusion_matrix(y_test, y_pred)

# predict (poorly calibrated) probabilities and calculate AUC
y_prob = nb.predict_proba(X_test_dtm)[:,1]
y_prob
print metrics.roc_auc_score(y_test, y_prob)



# exercise: show the message text for the false positives
#Match X_test and y_test
for index in range(len(X_test)):
    #if predicted is greater than actual it means you predicted spam when it was ham (and spam is positive: "tested positive for spam")
    if y_pred[index] > y_test[index]:
        print X_test[index]        
#Class answer using Pandas syntax        
X_test[np.where(y_pred > y_test)]        
X_test[y_pred > y_test]
#Class answer using Pandas syntax        


        

# exercise: show the message text for the false negatives
for index in range(len(X_test)):
    if y_pred[index] < y_test[index]:
        print X_test[index]
#Class answer using Pandas syntax        
X_test[y_pred < y_test]
#Class answer using Pandas syntax        



## COMPARE NAIVE BAYES AND LOGISTIC REGRESSION
## USING ALL DATA AND CROSS-VALIDATION

# create a document-term matrix using all data
all_dtm = vect.fit_transform(data.msg)

# instantiate logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# compare AUC using cross-validation
# note: this is slightly improper cross-validation... can you figure out why?
from sklearn.cross_validation import cross_val_score
cross_val_score(nb, all_dtm, data.label_bin, cv=10, scoring='roc_auc').mean()
cross_val_score(logreg, all_dtm, data.label_bin, cv=10, scoring='roc_auc').mean()



## EXERCISE: CALCULATE THE 'SPAMMINESS' OF EACH TOKEN

# create separate DataFrames for ham and spam
spammy = data[data.label=='spam']
hammy = data[data.label=='ham']

# learn the vocabulary of ALL messages and save it
vocab_dtm = vect.fit_transform(data.msg)


# create document-term matrix of ham, then convert to a regular array
# create document-term matrix of spam, then convert to a regular array
spammy_dtm = vect.transform(spammy)
hammy_dtm = vect.transform(hammy)

spammy_arr = spammy_dtm.toarray()
hammy_arr = hammy_dtm.toarray()



# count how many times EACH token appears across ALL messages in ham_arr
np.sum(hammy_arr, axis=0)

# count how many times EACH token appears across ALL messages in spam_arr
np.sum(spammy_arr, axis=0)

# create a DataFrame of tokens with their separate ham and spam counts
all_features = vect.get_feature_names()
spamham = pd.DataFrame({'token':all_features, 'spamcount':np.sum(spammy_arr, axis=0), 'hamcount': np.sum(hammy_arr,axis=0)})
np.sum(spamham.spamcount)
# add one to ham counts and spam counts so that ratio calculations (below) make more sense



# calculate ratio of spam-to-ham for each token






















































'''
CLASS: Naive Bayes SMS spam classifier
DATA SOURCE: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
'''

## READING IN THE DATA

# read tab-separated file using pandas
import pandas as pd
df = pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT5/master/data/SMSSpamCollection.txt',
                   sep='\t', header=None, names=['label', 'msg'])

# examine the data
df.head(20)
df.label.value_counts()
df.msg.describe()

# convert label to a binary variable
df['label'] = df.label.map({'ham':0, 'spam':1})
df.head()

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.msg, df.label, random_state=1)
X_train.shape
X_test.shape

## COUNTVECTORIZER: 'convert text into a matrix of token counts'
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

from sklearn.feature_extraction.text import CountVectorizer

# start with a simple example
train_simple = ['call you tonight',
                'Call me a cab',
                'please call me... PLEASE!']

# learn the 'vocabulary' of the training data
vect = CountVectorizer()
vect.fit(train_simple)
vect.get_feature_names()

# transform training data into a 'document-term matrix'
train_simple_dtm = vect.transform(train_simple)
train_simple_dtm
train_simple_dtm.toarray()


# examine the vocabulary and document-term matrix together
pd.DataFrame(train_simple_dtm.toarray(), columns=vect.get_feature_names())

# transform testing data into a document-term matrix (using existing vocabulary)
test_simple = ["please don't call me"]
test_simple_dtm = vect.transform(test_simple)
test_simple_dtm.toarray()
pd.DataFrame(test_simple_dtm.toarray(), columns=vect.get_feature_names())


## REPEAT PATTERN WITH SMS DATA

# instantiate the vectorizer
vect = CountVectorizer()

# learn vocabulary and create document-term matrix in a single step
train_dtm = vect.fit_transform(X_train)
train_dtm

# transform testing data into a document-term matrix
test_dtm = vect.transform(X_test)
test_dtm


# store feature names and examine them
train_features = vect.get_feature_names()
len(train_features)
train_features[:50]
train_features[-50:]

# convert train_dtm to a regular array
train_arr = train_dtm.toarray()
train_arr

## SIMPLE SUMMARIES OF THE TRAINING DATA

# refresher on NumPy
import numpy as np
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
arr
arr[0, 0]
arr[1, 3]
arr[0, :]
arr[:, 0]
np.sum(arr)
np.sum(arr, axis=0)
np.sum(arr, axis=1)


# exercise: calculate the number of tokens in the 0th message in train_arr

pd.DataFrame(train_arr, columns=vect.get_feature_names())
np.sum(train_arr, axis=0)

train_arr[0,:].sum()

np.sum(train_arr[0])

# exercise: count how many times the 0th token appears across ALL messages in train_arr
train_arr[:,0].sum()

np.sum(train_arr[:,0])


# exercise: count how many times EACH token appears across ALL messages in train_arr

np.sum(train_arr, axis=0)



# exercise: create a DataFrame of tokens with their counts
sums = list(np.sum(train_arr, axis=0))
zipped = zip(train_features, sums)
pd.DataFrame(zipped)


df = pd.DataFrame({'token':train_features, 'count':np.sum(train_arr, axis=0)})


## MODEL BUILDING WITH NAIVE BAYES
## http://scikit-learn.org/stable/modules/naive_bayes.html

# train a Naive Bayes model using train_dtm
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(train_dtm, y_train)

# make predictions on test data using test_dtm
y_pred = nb.predict(test_dtm)
y_pred

# compare predictions to true labels
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred)
print metrics.confusion_matrix(y_test, y_pred)

# predict (poorly calibrated) probabilities and calculate AUC
y_prob = nb.predict_proba(test_dtm)[:, 1]
y_prob
print metrics.roc_auc_score(y_test, y_prob)


# exercise: show the message text for the false positives


# exercise: show the message text for the false negatives
## COMPARE NAIVE BAYES AND LOGISTIC REGRESSION
## USING ALL DATA AND CROSS-VALIDATION

# create a document-term matrix using all data
all_dtm = vect.fit_transform(df.msg)

# instantiate logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# compare AUC using cross-validation
# note: this is slightly improper cross-validation... can you figure out why?
from sklearn.cross_validation import cross_val_score
cross_val_score(nb, all_dtm, df.label, cv=10, scoring='roc_auc').mean()
cross_val_score(logreg, all_dtm, df.label, cv=10, scoring='roc_auc').mean()
