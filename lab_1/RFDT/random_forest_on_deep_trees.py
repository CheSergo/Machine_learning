import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset from CSV file
data = pd.read_csv('../bioresponse.csv')

# Select only the first column as the target variable
target_variable = data.iloc[:, 0]

# Select all other columns as features for training and testing
features = data.iloc[:, 1:]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target_variable, test_size=0.2)

# # Create a random forest classifier with 50 trees and max depth of 10
# clf = RandomForestClassifier(n_estimators=50, max_depth=10)
# Create a random forest classifier with 10 deep trees (for example)
clf = RandomForestClassifier(n_estimators=10,max_depth=10)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# # Make predictions on the testing data
# y_pred = clf.predict(X_test)
# Make predictions on the testing data and get predicted probabilities
y_prob = clf.predict_proba(X_test)[:, 1]

# Convert predicted probabilities to binary predictions based on threshold value of 0.5 (for example)
y_pred = (y_prob >= 0.5).astype(int)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision score
precision = precision_score(y_test, y_pred)

# Calculate recall score
recall = recall_score(y_test, y_pred)

# Calculate F1-score
f1score = f1_score(y_test, y_pred)

# Calculate Log-loss score (requires predicted probabilities)
# y_prob = clf.predict_proba(X_test)
logloss = log_loss(y_test, y_prob)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1score)
print("Log-loss:", logloss)
print("========")
print(y_pred)

# Calculate precision-recall curve values
# precision, recall, thresholds = precision_recall_curve(y_test, y_prob[:, 1])
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Calculate ROC curve values
# fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot the precision-recall curve
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
# plt.show()
plt.savefig('RFDT_precision_recall_curve.png')

# Plot the ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.show()
plt.savefig('RFDT_roc_curve.png')