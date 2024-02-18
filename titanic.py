import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as sm
import csv

data = pd.read_csv(r'C:\Users\hp\Downloads\train (1).csv')
print(data.head())
print(data.duplicated())
data.drop_duplicates(inplace=True)
# We can see that there are no duplicates in the code
print(data.info())
# We see that Age, Cabin and embarked have uneven quantities

categorical_columns = [i for i in data.columns if data[i].dtype == 'object']
print(f'Categorical Columns: {categorical_columns}')
numerical_columns = [num for num in data.columns if data[num].dtype != 'object']
print(f'Numerical Columns: {numerical_columns}')

# Let's find the number of unique values in each column
print(data[categorical_columns].nunique())

# Machine learning doesn't understand categorical data so we can drop certain unneccesary columns
# We can drop Names since it's independent of everything. Let's check for Ticket since it's the most

print(data['Ticket'].unique()[:50])
# The ticket column has both numerical and categorical data so we drop it

data = data.drop(columns=['Name', 'Ticket'])
print(data.shape)

# Checking for percentage missing values in the rows
print(((round(data.isnull().sum())/data.shape[0])*100, 2))
# print((data.isnull().sum()/data.shape[0])*100)

# Cabin has too many missing values so we drop
data = data.drop(columns=['Cabin'])
# For Embarked column, we only remove the missing rows
data.dropna(subset = ['Embarked'], axis = 0, inplace = True)
print(data.shape)
# For Age, we impute missing values with the mean
# data = data.fillna(data.Age.mean())
data['Age'] = data['Age'].fillna(data['Age'].mean())
print(data.isnull().sum())

# Handling Outliers
# There are outliers in the values < 5 and those > 55
# Using the quartiles is even better
data['Age'].values[data['Age'].values>55]=55
data['Age'].values[data['Age'].values<3]=3
# data = data[~(data['Age'].values>50)]
# data = data[~(data['Age'].values<3)]
plt.boxplot(data['Age'], vert=False)
plt.ylabel('Variable')
plt.xlabel('Age')
plt.title('Box Plot')
plt.show()

data['Fare'].values[data['Fare'].values>60]=60
# data = data[~(data['Fare'].values>60)]

data['SibSp'].values[data['SibSp'].values>2]=2
# PassengerID, Survived, Pclass, SibSp, Parch, Fare, Embarked, Sex
plt.rcParams['figure.figsize'] = (16, 4)

plt.subplot(1, 6, 1)
sns.boxplot(data['PassengerId'])

plt.subplot(1, 6, 2)
sns.boxplot(data['Survived'])

plt.subplot(1, 6, 3)
sns.boxplot(data['Pclass'])

plt.subplot(1, 6, 4)
sns.boxplot(data['SibSp'])

plt.subplot(1, 6, 5)
sns.boxplot(data['Parch'])

plt.subplot(1, 6, 6)
sns.boxplot(data['Fare']).set_xticklabels(sns.boxplot(data['Fare']).get_xticklabels(), rotation=45)

plt.show()
# Alright. No more outliers detected

# Survived is our target variable so everything else will be used for training except PassengerId
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Using the preprocessing tool for scaling
# Some small feature engineering
# Always do this after dealing with outliers since feature scaling is very sensitive to outliers
scaler = MinMaxScaler(feature_range=(0,1))
numerical_columns_ = [col for col in X.columns if X[col].dtype != 'object']
X[numerical_columns_] = scaler.fit_transform(X[numerical_columns_])
print(X.head())
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], dtype=int)
# The dummy variables created could lead to a trap so always exclude on of the dummy variables 
# when training the model

# We're done cleaning our training dataset
# Next let's test our data by splitting it

# Using Stratified k fold Cross Validation
# X = pd.DataFrame(X).sample(frac=1).reset_index(drop=True) # This drastically reduced accuracy
print(X.head())
skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
for train, test in skf.split(X, y):
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]

# Using Random Forest Algorithm for prediction
rf_classifier = RandomForestClassifier(random_state=1)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
print(f'\n\nrf_predictions:{rf_predictions}')
print(f'\nrf_accuracy score: {rf_accuracy}')
print('\nrecall score',rf_recall)
print('\nf1 score',rf_f1)

# Using Decision Tree
# decision_tree = DecisionTreeClassifier(max_depth=3)
# decision_tree.fit(X_train, y_train)
# dect_predictions = decision_tree.predict(X_test)
# dect_accuracy = accuracy_score(y_test, dect_predictions)
# print(f'\ndect_predictions: {dect_predictions}')
# print(f'\ndect_accuracy: {dect_accuracy}')

# Trying linear regression
# lin_reg = LinearRegression()
# lin_reg.fit(X_train, y_train)
# lin_predictions = lin_reg.predict(X_test)
# lin_accuracy = sm.r2_score(y_test, lin_predictions)
# print(f'\nlin_predictions: {lin_predictions}')
# print(f'\nlin_accuracy: {lin_accuracy}')

grid = {'n_estimators': [25, 50, 100, 150],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9, 30],
        'max_leaf_nodes': [3, 6, 9, 30]}

# Hyperparameter tuning using GridSearchCV
# grid_search = GridSearchCV(RandomForestClassifier(), param_grid=grid)
# grid_search.fit(X_train, y_train)
# print(grid_search.best_estimator_)

# Updating the model
model_grid = RandomForestClassifier(max_depth = 30, max_features = None,
                                    max_leaf_nodes=30, n_estimators=50, random_state=1)
model_grid.fit(X_train, y_train)
model_grid_predict = model_grid.predict(X_test)
model_accuracy = accuracy_score(y_test, model_grid_predict)
model_recall = recall_score(y_test, model_grid_predict)
model_f1 = f1_score(y_test, model_grid_predict)
print('\n\nrf_new_accuracy score',model_accuracy)
print('\nnew_recall score',model_accuracy)
print('\nnew_f1 score',model_accuracy)
print(f'\nX train: {X_train.shape}')
print(f'y train: {y_train.shape}')
print(f'X test: {X_test.shape}')
print(f'y test: {y_test.shape}')

# Now let's work on our actual test data
test_data = pd.read_csv(r'C:\Users\hp\Downloads\test.csv')
print(test_data.head())
print(test_data.duplicated())
test_data.drop_duplicates(inplace=True)

print(test_data.info())
# Only Age, Cabin and Fare don't make up the 418. Probably have missing values

test_numerical_columns = [col for col in test_data.columns if test_data[col].dtype != 'object']
test_categorical_columns = [col for col in test_data.columns if test_data[col].dtype == 'object']
print(f'Categorical Columns: {test_categorical_columns}')
print(f'Numerical Columns: {test_numerical_columns}')
print(test_data[test_categorical_columns].nunique())

# Let's drop the Name column and check the Ticket data
print(test_data['Ticket'].unique()[:50])
# There's a mixture of objects and floats so we drop the Ticket column as well

test_data = test_data.drop(columns=['Name', 'Ticket'])

# Now let's check for missing values in the rows
print((test_data.isnull().sum()/test_data.shape[0])*100)
# We have some missing values in the Age, Fare and Cabin rows
# Cabin has 78 percent missing values so we drop it
test_data = test_data.drop(columns=['Cabin'])
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())

# After dealing with missing values let's see
print((test_data.isnull().sum()/test_data.shape[0])*100)
print(test_data.shape)
# Perfect

# Next we remove outliers in the Age and Fare data
test_data['Age'].values[test_data['Age'].values>55]=55
test_data['Age'].values[test_data['Age'].values<10]=10
# test_data = test_data[~(test_data['Age'].values>55)]
# test_data = test_data[~(test_data['Age'].values<10)]

test_data['SibSp'].values[test_data['SibSp'].values>2]=2

test_data['Fare'].values[test_data['Fare'].values>65]=65
# test_data = test_data[~(test_data['Fare'].values>65)]

print(test_data.shape)
plt.rcParams['figure.figsize'] = (16,4)
print(test_data.head())

plt.subplot(1, 5, 1)
sns.boxplot(test_data['Pclass'])

plt.subplot(1, 5, 2)
sns.boxplot(test_data['Age'])
 
plt.subplot(1, 5, 3)
sns.boxplot(test_data['SibSp'])

plt.subplot(1, 5, 4)
sns.boxplot(test_data['Parch'])

plt.subplot(1, 5, 5)
sns.boxplot(test_data['Fare'])

plt.show()

print(test_data.shape)

# PassengerId is irrelevant so we take that out
# Then we make Sex and Embarked into dummy variables

PassengerId = test_data['PassengerId']
test_data = test_data.drop(columns=['PassengerId'])
test_scaler = MinMaxScaler(feature_range=(0,1))
test_numerical_columns_ = [col for col in test_data.columns if test_data[col].dtype != 'object']
test_data[test_numerical_columns_] = test_scaler.fit_transform(test_data[test_numerical_columns_])
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], dtype = int)
print(test_data.head())
# The actual test data is ready

# Finally let's use our model on the test data
test_data_predictions = model_grid.predict(test_data)
test_data_predictions = pd.DataFrame(test_data_predictions)
test_data_predictions.columns= ['Survived']
test_data_predictions['PassengerId'] = PassengerId
column_titles = ["PassengerId", "Survived"]
test_data_predictions = test_data_predictions.reindex(columns=column_titles)
test_data_predictions['PassengerId'] = test_data_predictions['PassengerId'].fillna(0).astype(int)
print(test_data_predictions)
test_data_predictions.to_csv(r'C:\Users\hp\Desktop\Titanic1.csv', index=False)