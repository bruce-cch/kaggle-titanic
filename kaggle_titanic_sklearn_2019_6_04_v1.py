# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:05:25 2019

@author: Bruce.Chen
"""
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

#train_df = pd.read_csv('~/Downloads/pythonwork/kaggle/titanic/train.csv')
#test_df = pd.read_csv('~/Downloads/pythonwork/kaggle/titanic/test.csv')
train_df = pd.read_csv('C:/pythonwork/train.csv')
test_df = pd.read_csv('C:/pythonwork/test.csv')

#train_df = pd.read_csv('../input/train.csv')
#test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

print(train_df.columns.values)

# preview the data
train_df.head()
train_df.tail()

"""
Which features contain blank, null or empty values?

These will require correcting.

Cabin > Age > Embarked features contain a number of null values in that order
 for the training dataset.
Cabin > Age are incomplete in case of test dataset.
What are the data types for various features?

Helping us during converting goal.

Seven features are integer or floats. Six in case of test dataset.
Five features are strings (object).
"""
train_df.info()
print('_'*40)
test_df.info()
"""
What is the distribution of numerical feature values across the samples?

This helps us determine, among other early insights, how representative is the
 training dataset of the actual problem domain.

Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).
Survived is a categorical feature with 0 or 1 values.
Around 38% samples survived representative of the actual survival rate at 32%.
Most passengers (> 75%) did not travel with parents or children.
Nearly 30% of the passengers had siblings and/or spouse aboard.
Fares varied significantly with few passengers (<1%) paying as high as $512.
Few elderly passengers (<1%) within age range 65-80.
"""

train_df.describe(include=['O'])

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()\
                                .sort_values(by='Survived', ascending=False)

train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()\
                                .sort_values(by='Survived', ascending=False)

train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()\
                               .sort_values(by='Survived', ascending=False)

train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()\
                               .sort_values(by='Survived', ascending=False)

"""
Analyze by visualizing data
"""
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()



print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

"""
We decide to retain the new Title feature for model training.

"""

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()

"""
Completing a numerical continuous feature
Now we should start estimating and completing features with missing or null values.
We will first do this for the Age feature.

We can consider three methods to complete a numerical continuous feature.

A simple way is to generate random numbers between mean and standard deviation.

More accurate way of guessing missing values is to use other correlated features.
 In our case we note correlation among Age, Gender, and Pclass. Guess Age values 
 using median values for Age across sets of Pclass and Gender feature combinations.
 So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...

Combine methods 1 and 2. So instead of guessing age values based on median, use 
random numbers between mean and standard deviation, based on sets of Pclass and
 Gender combinations.

Method 1 and 3 will introduce random noise into our models. The results from
 multiple executions might vary. We will prefer method 2.
"""
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()\
                                 .sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()\
                                    .sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()\
                                  .sort_values(by='Survived', ascending=False)


"""
Converting categorical feature to numeric
We can now convert the EmbarkedFill feature by creating a new numeric Port feature.
"""

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()

"""
Quick completing and converting a numeric feature
We can now complete the Fare feature for single missing value in test dataset
 using mode to get the value that occurs most frequently for this feature. We
 do this in a single line of code.

Note that we are not creating an intermediate new feature or doing any further
 analysis for correlation to guess missing feature as we are replacing only a
 single value. The completion goal achieves desired requirement for model
 algorithm to operate on non-null values.

We may also want round off the fare to two decimals as it represents currency
"""
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean()\
                                  .sort_values(by='FareBand', ascending=True)


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)

test_df.head(10)

"""
Model, predict and solve
Now we are ready to train a model and predict the required solution.
 There are 60+ predictive modelling algorithms to choose from. We must
 understand the type of problem and solution requirement to narrow down
 to a select few models which we can evaluate. Our problem is a classification
 and regression problem. We want to identify relationship between output
 (Survived or not) with other variables or features (Gender, Age, Port...).
 We are also perfoming a category of machine learning which is called
 supervised learning as we are training our model with a given dataset.
 With these two criteria - Supervised Learning plus Classification and
 Regression, we can narrow down our choice of models to a few. These include:

     
Logistic Regression
KNN or k-Nearest Neighbors
Support Vector Machines
Naive Bayes classifier
Decision Tree
Random Forrest
Perceptron
Artificial neural network
RVM or Relevance Vector Machine

"""

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

#KNN  KNN confidence score is better than Logistics Regression but worse than SVM.
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest



"""
#Model evaluation
We can now rank our evaluation of all the models to choose the best one for
 our problem. While both Decision Tree and Random Forest score the same,
 we choose to use Random Forest as they correct for decision trees' habit
 of overfitting to their training set.

	Model	Score
3	Random Forest	86.76
8	Decision Tree	86.76
1	KNN	84.74
0	Support Vector Machines	83.84
2	Logistic Regression	80.36
7	Linear SVC	79.12
6	Stochastic Gradient Decent	78.56
5	Perceptron	78.00
4	Naive Bayes	72.28

"""
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


submission = pd.DataFrame( {"PassengerId": test_df["PassengerId"],"Survived": Y_pred} )
# submission.to_csv('../output/submission.csv', index=False)


"""

param_grid = [ { 'criterion': [ 'gini' ],
                 'splitter': [ 'best', 'random' ],
                 'max_depth': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                 'min_samples_split': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                 'min_samples_leaf': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ] },
               { 'criterion': [ 'entropy' ],
                 'splitter': [ 'best', 'random' ],
                 'max_depth': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                 'min_samples_split': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                 'min_samples_leaf': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ] } ]
"""
#Optimization of Decision Tree
param_grid = [ { 'DT__criterion': [ 'gini' ],
                 'DT__splitter': [ 'best', 'random' ],
                 'DT__max_depth': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                 'DT__min_samples_split': [  2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                 'DT__min_samples_leaf': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ] },
               { 'DT__criterion': [ 'entropy' ],
                 'DT__splitter': [ 'best', 'random' ],
                 'DT__max_depth': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                 'DT__min_samples_split': [ 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                 'DT__min_samples_leaf': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ] } ]


#pipe = Pipeline([ ( 'DT', DecisionTreeClassifier() ) ])
#pipe.fit(X_train, Y_train)
#grid = GridSearchCV( pipe, param_grid=param_grid, cv=5 )
#grid.fit(X_train, Y_train)

#print( "The Best Cross-Validation Accuracy: {:.2f}".format( grid.best_score_ ) )
#print( 'The Best parameters of Decision Tree: {}'.format( grid.best_params_ ) )


"""
The Best Cross-Validation Accuracy: 0.82
{'DT__criterion': 'entropy',
 'DT__max_depth': 5,
 'DT__min_samples_leaf': 2,
 'DT__min_samples_split': 2,
 'DT__splitter': 'random'}
"""

decision_tree = DecisionTreeClassifier( criterion= 'entropy', max_depth= 5,
                                        min_samples_leaf = 2,
                                        min_samples_split = 2,
                                        splitter = 'random' )
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

submission = pd.DataFrame({ "PassengerId": test_df["PassengerId"], "Survived": Y_pred})
submission.to_csv('C:/pythonwork/titanic_sklearn_tree_2019_6_13.csv', index=False)




#Optimization of Random Forest
param_grid = [ { 'RF__criterion': [ 'gini' ],
                 'RF__max_depth': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                 'RF__min_samples_split': [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200 ],
                 'RF__min_samples_leaf': [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200 ],
                 'RF__n_estimators': [ 10, 50, 100, 150, 200, 250, 300 ] },
               { 'RF__criterion': [ 'entropy' ],
                 'RF__max_depth': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                 'RF__min_samples_split': [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200 ],
                 'RF__min_samples_leaf': [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200 ],
                 'RF__n_estimators': [ 10, 50, 100, 150, 200, 250, 300 ]} ]

"""

pipe = Pipeline([ ( 'RF', RandomForestClassifier() ) ])
#pipe.fit(X_train, Y_train)
Grid = GridSearchCV( pipe, param_grid=param_grid, cv=5, n_jobs=-1 )
Grid.fit(X_train, Y_train)

The Best Cross-Validation Accuracy of RF: 0.83
The Best parameters of RF: {'RF__criterion': 'entropy', 'RF__max_depth': 5,
'RF__min_samples_leaf': 2, 'RF__min_samples_split': 7, 'RF__n_estimators': 50}
print( "The Best Cross-Validation Accuracy of RF: {:.2f}".format( Grid.best_score_ ) )
print( 'The Best parameters of RF: {}'.format( Grid.best_params_ ) )
"""
random_forest = RandomForestClassifier( criterion= 'entropy',
                                        max_depth= 5,
                                        min_samples_leaf= 2,
                                        min_samples_split= 7,
                                        n_estimators= 50, n_jobs=-1 )
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

submission = pd.DataFrame({ "PassengerId": test_df["PassengerId"],
                            "Survived": Y_pred})

submission.to_csv('C:/pythonwork/titanic_sklearn_rf_2019_6_13.csv', index=False)



#Optimization of KNN
"""
param_grid = [ n_neighbors : [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ],
               weights : [ ‘uniform’ , ‘distance’  ],
               leaf_size : [ 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 ],
               p : [ 1, 2 ] ]


param_grid = [ { 'knn__algorithm' : [ 'ball_tree' ],
                 'knn__n_neighbors' : [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ],
                 'knn__weights' : [ 'uniform' , 'distance'  ],
                 'knn__leaf_size' : [ 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 ],
                 'knn__p' : [ 1, 2 ] },
               { 'knn__algorithm' : [ 'kd_tree' ],
                 'knn__n_neighbors' : [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ],
                 'knn__weights' : [ 'uniform' , 'distance'  ],
                 'knn__leaf_size' : [ 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 ],
                 'knn__p' : [ 1, 2 ]},
               { 'knn__algorithm' : [ 'brute' ],
                 'knn__n_neighbors' : [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ],
                 'knn__weights' : [ 'uniform' , 'distance'  ],
                 'knn__leaf_size' : [ 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 ],
                 'knn__p' : [ 1, 2 ]}  ]

pipe = Pipeline([ ( 'knn', KNeighborsClassifier( ) ) ])
Grid = GridSearchCV( pipe, param_grid=param_grid, cv=5, n_jobs=-1 )
Grid.fit(X_train, Y_train)
print( "The Best Cross-Validation Accuracy of KNN: {:.2f}".format( Grid.best_score_ ) )
print( 'The Best parameters of KNN: {}'.format( Grid.best_params_ ) )

The Best Cross-Validation Accuracy of KNN: 0.82
The Best parameters of KNN: {'knn__algorithm': 'ball_tree', 'knn__leaf_size': 5,
 'knn__n_neighbors': 10, 'knn__p': 1, 'knn__weights': 'uniform'}


"""



knn = KNeighborsClassifier( algorithm=  'ball_tree', leaf_size= 5,
                            n_neighbors= 10, p= 1, weights= 'uniform',
                            n_jobs=-1 )
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
submission = pd.DataFrame({ "PassengerId": test_df["PassengerId"],
                            "Survived": Y_pred})
submission.to_csv('C:/pythonwork/titanic_sklearn_knn_2019_6_13.csv', index=False)


#Optimization of Logistic Regression
"""
param_grid = [ { 'LGR__penalty': [ 'l1' ],
                 'LGR__C': [ 0.001, 0.01, 0.1, 1  ],
                 'LGR__solver': [  'liblinear',  'saga' ],
                 'LGR__max_iter': [ 100, 500, 600 ] },
               { 'LGR__penalty': [ 'l2' ],
                 'LGR__C': [ 0.001, 0.01, 0.1, 1  ],
                 'LGR__solver': [ 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga' ],
                 'LGR__max_iter': [ 100, 500, 600 ] } ]

pipe = Pipeline([ ( 'LGR', LogisticRegression() ) ])
Grid = GridSearchCV( pipe, param_grid=param_grid, cv=5, n_jobs=-1 )
Grid.fit(X_train, Y_train)
print( "The Best Cross-Validation Accuracy of Logistic: {:.2f}".format( Grid.best_score_ ) )
print( 'The Best parameters of Logistic: {}'.format( Grid.best_params_ ) )

The Best Cross-Validation Accuracy of Logistic: 0.81
The Best parameters of Logistic: {'LGR__C': 0.1, 'LGR__max_iter': 100,
                       'LGR__penalty': 'l2', 'LGR__solver': 'newton-cg'}

"""
logreg = LogisticRegression( C= 0.1, max_iter= 100, penalty= 'l2',
                             solver= 'newton-cg', n_jobs=-1 )
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
submission = pd.DataFrame({ "PassengerId": test_df["PassengerId"],
                            "Survived": Y_pred})
submission.to_csv('C:/pythonwork/titanic_sklearn_logistic_2019_6_13.csv', index=False)




#Optimization of Stochastic Gradient Descent
"""
param_grid = [{ 'sgd__penalty' : [ 'l2', 'l1', 'elasticnet' ],
                'sgd__alpha' : [ 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000 ],
                'sgd__max_iter' : [ 1000, 2000, 3000 ],
                'sgd__learning_rate' : [ 'constant', 'optimal', 'invscaling', 'adaptive' ],
                'sgd__power_t' : [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10, 100 ],
                'sgd__eta0' : [  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10, 100  ] }]

pipe = Pipeline([ ( 'sgd', SGDClassifier() ) ])
Grid = GridSearchCV( pipe, param_grid=param_grid, cv=5, n_jobs=-1 )
Grid.fit(X_train, Y_train)
print( "The Best Cross-Validation Accuracy of SGDClassifier: {:.2f}".format( Grid.best_score_ ) )
print( 'The Best parameters of SGDClassifier: {}'.format( Grid.best_params_ ) )



The Best Cross-Validation Accuracy of SGDClassifier: 0.81
The Best parameters of SGDClassifier: {'sgd__alpha': 0.0001, 'sgd__eta0': 0.9,
'sgd__learning_rate': 'invscaling', 'sgd__max_iter': 1000, 'sgd__penalty': 'l2', 'sgd__power_t': 0.9}
"""
sgd = SGDClassifier( alpha= 0.0001, eta0= 0.9, learning_rate= 'invscaling',
                     max_iter= 1000, penalty= 'l2', power_t= 0.9, n_jobs=-1 )
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

submission = pd.DataFrame({ "PassengerId": test_df["PassengerId"],
                            "Survived": Y_pred})
submission.to_csv('C:/pythonwork/titanic_sklearn_sgd_2019_6_13.csv', index=False)



#Optimization of Perceptron
"""
param_grid = [ { 'pcn__penalty':[ 'l2', 'l1', 'elasticnet' ],
                 'pcn__alpha': [ 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000 ],
                 'pcn__max_iter': [ 1000, 2000, 3000, 4000 ] }]

pipe = Pipeline([ ( 'pcn', Perceptron() ) ])
Grid = GridSearchCV( pipe, param_grid=param_grid, cv=5, n_jobs=-1 )
Grid.fit(X_train, Y_train)
print( "The Best Cross-Validation Accuracy of Perceptron: {:.2f}".format( Grid.best_score_ ) )
print( 'The Best parameters of Perceptron: {}'.format( Grid.best_params_ ) )


The Best Cross-Validation Accuracy of Perceptron: 0.76
The Best parameters of Perceptron: {'pcn__alpha': 0.001, 'pcn__max_iter': 1000, 'pcn__penalty': 'l1'}
"""
perceptron = Perceptron( alpha= 0.001, max_iter= 1000, penalty= 'l1', n_jobs=-1 )
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
submission = pd.DataFrame({ "PassengerId": test_df["PassengerId"],
                            "Survived": Y_pred})
submission.to_csv('C:/pythonwork/titanic_sklearn_perceptron_2019_6_13.csv', index=False)



























































































































































































