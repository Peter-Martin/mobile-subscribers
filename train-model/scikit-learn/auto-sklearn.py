# =============================================================================
# Automatic model training using scikit-learn and autosklearn package
# =============================================================================

import pandas as pd

from autosklearn.estimators import AutoSklearnClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# -----------------------------------------------------------------------------
# 1) Import training and test data sets, split each into features/labels

# training_features = pd.read_csv('../../prepare-data/one-label/training.csv')
# training_features = pd.read_csv('../../prepare-data/one-label/simple/training.csv')
training_features = pd.read_csv(
    '../../prepare-data/one-label/simple/downgrade/postpaid/training.csv')
training_labels = training_features.pop('UpdatedIn90Days').values

# test_features = pd.read_csv('../../prepare-data/one-label/test.csv')
# test_features = pd.read_csv('../../prepare-data/one-label/simple/test.csv')
test_features = pd.read_csv(
    '../../prepare-data/one-label/simple/downgrade/postpaid/test.csv')
test_labels = test_features.pop('UpdatedIn90Days').values

# -----------------------------------------------------------------------------
# 2) Fit auto-classifier

clf = AutoSklearnClassifier()
clf.fit(training_features, training_labels)

# -----------------------------------------------------------------------------
# 3) Perform predictions on test set

actual = test_labels
predictions = clf.predict(test_features)

# -----------------------------------------------------------------------------
# 4) Show result scores; confusion matrix (most useful) and precision/recall

print('\nconfusion matrix')
# print(confusion_matrix(actual, predictions, labels = [0, 1, 2, 3, 4]))
# print(confusion_matrix(actual, predictions, labels = [0, 1, 2]))
print(confusion_matrix(actual, predictions, labels = [0, 1]))
print('\nprecision')
# print(precision_score(actual, predictions, labels = [0, 1, 2, 3, 4], average = 'weighted'))
# print(precision_score(actual, predictions, labels = [0, 1, 2], average = 'weighted'))
print(precision_score(actual, predictions, labels = [0, 1], average = 'weighted'))
print('\nrecall')
# print(recall_score(actual, predictions, labels = [0, 1, 2, 3, 4], average = 'weighted'))
# print(recall_score(actual, predictions, labels = [0, 1, 2], average = 'weighted'))
print(recall_score(actual, predictions, labels = [0, 1], average = 'weighted'))
