Bank Deposit Prediction using Decision Tree Classifier
This script uses a Decision Tree Classifier to predict whether a customer will make a deposit based on their demographic and financial information. The data is sourced from the bank dataset.

Prerequisites
->Python 3.x
->Pandas library
->Scikit-learn library
->Matplotlib library

Script:
#Task-3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# Read csv file
pima = pd.read_csv(r'C:\dbms\prodigy (2).csv')

# Define feature columns and target
feature_cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
X = pima[feature_cols]
y = pima['deposit']

# One-hot encoding of categorical variables
X_encoded = pd.get_dummies(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=1)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X_train, y_train)

# Predicting the outcome for the test set
y_pred = clf.predict(X_test)

# Calculate and print the accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plotting the decision tree
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=X_encoded.columns.tolist(), class_names=['0', '1'])
plt.show()

Description:
This script performs the following tasks:

Import Libraries:

Imports necessary libraries for data processing (pandas), machine learning (scikit-learn), and plotting (matplotlib).
Load Data:

Loads the dataset from the specified file path.
Prepare Data:

Defines feature columns and target variable.
Applies one-hot encoding to convert categorical variables into numeric format.
Split Data:

Splits the dataset into training and testing sets using train_test_split.
Train Model:

Initializes a DecisionTreeClassifier with a maximum depth of 3.
Trains the model on the training data.
Make Predictions:

Uses the trained model to predict outcomes for the test set.
Evaluate Model:

Calculates and prints the accuracy score, confusion matrix, and classification report.
Plot Decision Tree:

Plots the decision tree using plot_tree and displays it.

Output:
Running this script will output the accuracy of the model, the confusion matrix, and the classification report. Additionally, it will display a plot of the decision tree.
![acc](https://github.com/Lavanya-1234198/PRODIGY_DS_03/assets/174336088/8d0ee71d-ba3a-48b5-b49b-77b764e11935)
![tree](https://github.com/Lavanya-1234198/PRODIGY_DS_03/assets/174336088/b97b584d-404f-4aba-b999-0ba4a723ae35)
