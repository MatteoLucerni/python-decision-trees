import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

titanic = pd.read_csv("http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
titanic.info()

titanic = titanic.drop('Name', axis=1)


titanic = pd.get_dummies(titanic)
titanic.head()

X = titanic.drop('Survived', axis=1).values
Y = titanic['Survived'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

dt = DecisionTreeClassifier(criterion='gini')
dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)

acc = accuracy_score(Y_test, Y_pred)

# check overfitting
Y_pred_train = dt.predict(X_train)

acc_train = accuracy_score(Y_train, Y_pred_train)

print(f'Accuracy - test: {acc} / train: {acc_train}')

# fix overfitting setting max depth (reducing model complexity)

for n in range(2,9):
    print('max depht: ' + str(n))
    dt = DecisionTreeClassifier(criterion='gini', max_depth=n)
    dt.fit(X_train, Y_train)
    Y_pred = dt.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)

    # check overfitting
    Y_pred_train = dt.predict(X_train)

    acc_train = accuracy_score(Y_train, Y_pred_train)

    print(f'Accuracy - test: {acc} / train: {acc_train}')
    print('=' * 50)

# visual graph
dt = DecisionTreeClassifier(criterion='gini', max_depth=6)
dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)

Y_pred_train = dt.predict(X_train)
acc_train = accuracy_score(Y_train, Y_pred_train)
print(f'Accuracy - test: {acc} / train: {acc_train}')
print('=' * 50)

dotfile = open('tree.dot', 'w')
export_graphviz(dt, out_file= dotfile, feature_names= titanic.columns.drop('Survived'))
dotfile.close()

