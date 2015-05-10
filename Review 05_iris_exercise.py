'''
EXERCISE: "Human Learning" with iris data

Can you predict the species of an iris using petal and sepal measurements?

TASKS:
1. Read iris data into a pandas DataFrame, including column names.
2. Gather some basic information about the data.
3. Use groupby, sorting, and/or plotting to look for differences between species.
4. Come up with a set of rules that could be used to predict species based upon measurements.

BONUS: Define a function that accepts a row of data and returns a predicted species.
Then, use that function to make predictions for all existing rows of data.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## TASK 1

# read the iris data into a pandas DataFrame, including column names
col_names = ['slength', 'swidth', 'plength', 'pwidth', 'species']
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=col_names)



# TASK 2
# gather basic information     
df.isnull().sum()           #This is really important
df.shape
df.head()
df.describe()               #This is really useful
df.species.value_counts()   #Also very useful
df.dtypes                   #Necessary to know because... errors
                   


## TASK 3
# use groupby to look for differences between the species
df.groupby('species').slength.mean()
df.groupby('species').mean()
#Significant diff betw all 3 for mean slength, plength, and pwidth
df.groupby(df.species).describe() 
#1) By looking at Min/Max we can see no overlap in plength for Setosa, and Versicolor,and some overlap between versicolor and virginica.
#2) By look at the same for pwidth: no overlap betw Seto and Vers, and some overlap betw Vers and Virg
#3) By looking at slength we see overlap for all
#4) By looking at swidth we see overlap for all
#5) Summary: It's easy to distinguish Seto, but not easy to separate Vers and Virg


# use sorting to look for differences between the species
df.sort_index(by='slength').values
df.sort_index(by='swidth').values
df.sort_index(by='plength').values
df.sort_index(by='pwidth').values



# use plotting to look for differences between the species
df.plength.hist(by=df.species, sharex=True, sharey=True)
df.hist(column='plength', by='species', layout=(1,3), sharex=True, sharey=True)
df.hist(column='pwidth', by='species', layout=(1,3), sharex=True, sharey=True)
#Alternate Syntax: df.pwidth.hist(by=df.species, layout=(1,3), sharex=True, sharey=True)
df.boxplot(column='plength', by='species') #Confirms Min/Max info
df.boxplot(by='species') #Confirms Min/Max for all.  However it also appears that we can say that the min for Virg is always above the mean for Vers



df[df.species=='Iris-versicolor'].pwidth.mean() #1.326
df[df.species=='Iris-virginica'].pwidth.min() #1.400
#We can see that all pwidth for Virg are all larger than the mean pwidth for Vers
df[df.species=='Iris-versicolor'].plength.mean() #4.260
df[df.species=='Iris-virginica'].plength.min() #4.5
#We cannot say the same for plength
#Summary: 
#1) So we can tell Seto and Vers apart by pwidth and plength
#2) And we can tell if something is Virg if it's pwidth is larger than the mean pwidth for Vers



# map species to a numeric value so that plots can be colored by category
df['species_num'] = df.species.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
df.plot(kind='scatter', x='plength', y='pwidth', c='species_num', colormap='cubehelix')
#This again shows separability for Seto, and slight overlap for Vers and Virg for plength, and pwidth
pd.scatter_matrix(df, c=df.species_num)


## TASK 4
# If petal length is less than 3, predict setosa.
# Else if petal width is less than 1.8, predict versicolor.
# Otherwise predict virginica.
'''USING PLENGTH AS PER CLASS'''
'''
def ClassifyIris(row):
    if row[2] < 3: #If 2nd column in row is > 3
        return 0
    elif row[3] < 1.8:
        return 1
    else:
        return 2
'''        
'''USING PLENGTH AS PER CLASS'''



'''USING MEAN IN THE CALCULATION.  IS THIS OVERFITTING???'''
def ClassifyIris(row):
    if row[2] < 3:
        return 0
    elif row[3] < 1.8:
        return 1
    else:
        return 2
'''USING MEAN IN THE CALCULATION.  IS THIS OVERFITTING???'''

# predict for a single row
ClassifyIris(df.iloc[0,:]) #First row
ClassifyIris(df.iloc[-1, :]) #Last row

# store predictions for all rows
predictions = [ClassifyIris(row) for row in df.values]

# calculate the percentage of correct predictions
np.mean(df.species_num==predictions)





'''Class Answer'''
'''
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                   names=col_names)
                   
iris.shape
iris.head()
iris.describe()
iris.species.value_counts()
iris.dtypes
iris.isnull().sum()

iris.groupby('species').sepal_length.mean()
iris.groupby('species').mean()
iris.groupby('species').describe()

iris.sort_index(by='sepal_length').values
iris.sort_index(by='sepal_width').values
iris.sort_index(by='petal_length').values
iris.sort_index(by='petal_width').values

# use plotting to look for differences between the species
iris.petal_width.hist(by=iris.species, sharex=True)
iris.boxplot(column='petal_width', by='species')
iris.boxplot(by='species')

# map species to a numeric value so that plots can be colored by category
iris['species_num'] = iris.species.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
iris.plot(kind='scatter', x='petal_length', y='petal_width', c='species_num', colormap='Blues')
pd.scatter_matrix(iris, c=iris.species_num)

## BONUS
# define function that accepts a row of data and returns a predicted species
def classify_iris(row):
    if row[2] < 3:          # petal_length
        return 0    # setosa
    elif row[3] < 1.8:      # petal_width
        return 1    # versicolor
    else:
        return 2    # virginica
        
# predict for a single row
classify_iris(iris.iloc[0, :])      # first row
classify_iris(iris.iloc[149, :])    # last row

# store predictions for all rows
predictions = [classify_iris(row) for row in iris.values] 

# calculate the percentage of correct predictions
np.mean(iris.species_num == predictions)    # 0.96       
'''