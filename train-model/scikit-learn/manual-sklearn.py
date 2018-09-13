# =============================================================================
# Manual model training using scikit-learn
# =============================================================================

import pandas as pd

# note: some imports are only required when using the options later on,
# e.g., scaling, balancing, etc.

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score #, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# -----------------------------------------------------------------------------
# 1) Import training and test data sets, split each into features/labels

# training_features = pd.read_csv('/home/peter/Documents/projects/MobileSubscribers/atm/training.csv')
# training_features = pd.read_csv('/home/peter/Documents/projects/MobileSubscribers/atm/simple/training.csv')
# training_features = pd.read_csv('/home/peter/Documents/projects/MobileSubscribers/atm/simple/downgrade/training.csv')
training_features = pd.read_csv(
    '/home/peter/Documents/projects/MobileSubscribers/atm/simple/downgrade/postpaid/training.csv')
training_labels = training_features.pop('UpdatedIn90Days').values

# test_features = pd.read_csv('/home/peter/Documents/projects/MobileSubscribers/atm/test.csv')
# test_features = pd.read_csv('/home/peter/Documents/projects/MobileSubscribers/atm/simple/test.csv')
# test_features = pd.read_csv('/home/peter/Documents/projects/MobileSubscribers/atm/simple/downgrade/test.csv')
test_features = pd.read_csv(
    '/home/peter/Documents/projects/MobileSubscribers/atm/simple/downgrade/postpaid/test.csv')
test_labels = test_features.pop('UpdatedIn90Days').values

# -----------------------------------------------------------------------------
# 2) OPTIONALLY scale features, needed for SGD and others

# scaler = StandardScaler()
# scaler.fit(training_features)
# training_features = scaler.transform(training_features)
# test_features = scaler.transform(test_features)

# -----------------------------------------------------------------------------
# 3) OPTIONALLY apply SMOTE over/underbalancing to training set

# oversampler = SMOTE(random_state = 0)
# training_features, training_labels = oversampler.fit_sample(training_features, training_labels)

# -----------------------------------------------------------------------------
# 4) OPTIONALLY use kernel approximation

# rbf = RBFSampler(gamma = 1, random_state = 1)
# training_features = rbf.fit_transform(training_features)
# test_features = rbf.fit_transform(test_features)

# -----------------------------------------------------------------------------
# 5) Fit classifier

clf = RandomForestClassifier(random_state = 0)
# clf = RandomForestClassifier(random_state = 0, class_weight = {0:4, 1:32, 2:4, 3:1, 4:8})
# clf = GradientBoostingClassifier(random_state = 0)
# clf = SGDClassifier(loss = "hinge", penalty = "l2")
# clf = LinearSVC(loss='hinge', penalty="l2")
# pipeline with feature selection, useful for some classifiers
# clf = Pipeline([
#     #('feature_selection', SelectFromModel(LinearSVC(loss='hinge', penalty="l2"))),
#     ('feature_selection', SelectFromModel(GradientBoostingClassifier(random_state = 0))),
#     ('classification', GradientBoostingClassifier(random_state = 0))
# ])
clf.fit(training_features, training_labels)

# -----------------------------------------------------------------------------
# 6) OPTIONALLY cross-validate, usefulness varies per classifier

# scores = cross_val_score(clf, training_features, training_labels, cv = 3)
# print('CV score')
# print(scores)

# -----------------------------------------------------------------------------
# 7) Perform predictions on test set

actual = test_labels
predictions = clf.predict(test_features)

# catch cases where some classifiers never predict particular values
missing_predictions = set(actual) - set(predictions)
print('\nmissing predictions:')
print(missing_predictions)

# -----------------------------------------------------------------------------
# 8) Show result scores; confusion matrix (most useful) and precision/recall

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

# -----------------------------------------------------------------------------
# 9) OPTIONALLY plot curves; usefulness varies per classifier

# false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
# roc_auc = auc(false_positive_rate, true_positive_rate)
# print(roc_auc)
# plt.title('Receiver Operating Characteristic')
# plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0,1],[0,1],'r--')
# plt.xlim([-0.1,1.2])
# plt.ylim([-0.1,1.2])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')