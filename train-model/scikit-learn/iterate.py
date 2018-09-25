# =============================================================================
# Iteratively train model combinations using scikit-learn,
# i.e., by iteratively varying the following:
#   - Classifier
#   - Data Set (full / simplified / postpaid-only, etc.)
#   - Optionally scaling the data set
#   - Optionally oversampling the data set
# =============================================================================

import pandas as pd
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# -----------------------------------------------------------------------------

# builds various training/test datasets,
# also stores labels we wish to measure at prediction time

class Datasets:
    def __init__(self,
                 training_features,
                 test_features,
                 measured_labels):
        # build training/test features and labels
        self.training_features = pd.read_csv(training_features)
        self.training_labels = self.training_features.pop('UpdatedIn90Days').values
        self.test_features = pd.read_csv(test_features)
        self.test_labels = self.test_features.pop('UpdatedIn90Days').values
        self.measured_labels = measured_labels
        # build scaled training/test feature sets
        scaler = StandardScaler()
        scaler.fit(self.training_features)
        self.training_features_scaled = scaler.transform(self.training_features)
        self.test_features_scaled = scaler.transform(self.test_features)
        # build oversampled training set
        oversampler = SMOTE(random_state = 0)
        self.training_features_oversampled, self.training_labels_oversampled =\
            oversampler.fit_sample(self.training_features, self.training_labels)
        # also build combination: scaled + oversampled training set
        self.training_features_scaled_oversampled, self.training_labels_scaled_oversampled = \
            oversampler.fit_sample(self.training_features_scaled, self.training_labels)
        # store various sets in a dictionary for later lookup by get() method,
        # key is tuple of scaled and oversampled booleans
        self.dataset_table = {
            # neither scaled nor oversampled
            tuple([False, False]):  [
                self.training_features, self.training_labels,
                self.test_features, self.test_labels
            ],
            # just oversampled
            tuple([False, True]):   [
                self.training_features_oversampled, self.training_labels_oversampled,
                self.test_features, self.test_labels
            ],
            # just scaled
            tuple([True, False]):   [
                self.training_features_scaled, self.training_labels,
                self.test_features_scaled, self.test_labels
            ],
            # both scaled and oversampled
            tuple([True, True]):    [
                self.training_features_scaled_oversampled, self.training_labels_scaled_oversampled,
                self.test_features_scaled, self.test_labels
            ]
        }
    # return correct training/test sets according to scaled/oversampled booleans
    def get(self, scaled, oversampled):
        return self.dataset_table.get(tuple([scaled, oversampled]))

# we use ONE of the following data sets

# LABELS: multiple downgrade + upgrade
# SUBSCRIBERS: postpaid + prepaid
full = Datasets(
    '/home/peter/Documents/projects/MobileSubscribers/atm/training.csv',
    '/home/peter/Documents/projects/MobileSubscribers/atm/test.csv',
    [0, 1, 2]
)

# LABELS: single downgrade + upgrade
# SUBSCRIBERS: postpaid + prepaid
simple = Datasets(
    '/home/peter/Documents/projects/MobileSubscribers/atm/simple/training.csv',
    '/home/peter/Documents/projects/MobileSubscribers/atm/simple/test.csv',
    [0]
)

# LABELS: single downgrade
# SUBSCRIBERS: postpaid + prepaid
simple_downgrade = Datasets(
    '/home/peter/Documents/projects/MobileSubscribers/atm/simple/downgrade/training.csv',
    '/home/peter/Documents/projects/MobileSubscribers/atm/simple/downgrade/test.csv',
    [0]
)

# LABELS: single downgrade
# SUBSCRIBERS: postpaid
simple_downgrade_postpaid = Datasets(
    '/home/peter/Documents/projects/MobileSubscribers/atm/simple/downgrade/postpaid/training.csv',
    '/home/peter/Documents/projects/MobileSubscribers/atm/simple/downgrade/postpaid/test.csv',
    [0]
)

# -----------------------------------------------------------------------------

# we use ONE of these classifiers,
# note that we use a fresh classifier instance per training/prediction iteration

def mlp():
    return MLPClassifier(random_state = 0, hidden_layer_sizes = (10, 10))

def rf():
    return RandomForestClassifier(random_state = 0)

def gb():
    return GradientBoostingClassifier(random_state = 0)

def sgd():
    return SGDClassifier(loss = 'hinge', penalty = 'l2')

def lsv():
    return LinearSVC(loss = 'hinge', penalty= 'l2')

# -----------------------------------------------------------------------------

# define each iteration: classifier, data set, optionally scale/oversample

class Iteration:
    def __init__(self, name, datasets, scaled, oversampled, classifier):
        self.name = name
        self.datasets = datasets
        self.scaled = scaled
        self.oversampled = oversampled
        self.classifier = classifier

iterations = [

    # NB: Gradient Boosting with scaling/overbalancing doesn't run successfully,
    # so commented out those iterations below

    # iteration name, features/labels, scaled, oversampled, classifier
    
    # Iteration('Full | MLP', full, False, False, mlp()),
    # Iteration('Full | Random Forest', full, False, False, rf()),
    # Iteration('Full | Gradient Boosting', full, False, False, gb()),
    # Iteration('Full | SGD', full, False, False, sgd()),
    # Iteration('Full | Linear SVC', full, False, False, lsv()),
    # Iteration('Full | MLP | Oversampled', full, False, True, mlp()),
    # Iteration('Full | Random Forest | Oversampled', full, False, True, rf()),
    # # Iteration('Full | Gradient Boosting | Oversampled', full, False, True, gb()),
    # Iteration('Full | SGD | Oversampled', full, False, True, sgd()),
    # Iteration('Full | Linear SVC | Oversampled', full, False, True, lsv()),
    # Iteration('Full | MLP | Scaled', full, True, False, mlp()),
    # Iteration('Full | Random Forest | Scaled', full, True, False, rf()),
    # # Iteration('Full | Gradient Boosting | Scaled', full, True, False, gb()),
    # Iteration('Full | SGD | Scaled', full, True, False, sgd()),
    # Iteration('Full | Linear SVC | Scaled', full, True, False, lsv()),
    # Iteration('Full | MLP | Scaled | Oversampled', full, True, True, mlp()),
    # Iteration('Full | Random Forest | Scaled | Oversampled', full, True, True, rf()),
    # # Iteration('Full | Gradient Boosting | Scaled | Oversampled', full, True, True, gb()),
    # Iteration('Full | SGD | Scaled | Oversampled', full, True, True, sgd()),
    # Iteration('Full | Linear SVC | Scaled | Oversampled', full, True, True, lsv()),
    #
    # Iteration('Simple | MLP', simple, False, False, mlp()),
    # Iteration('Simple | Random Forest', simple, False, False, rf()),
    # Iteration('Simple | Gradient Boosting', simple, False, False, gb()),
    # Iteration('Simple | SGD', simple, False, False, sgd()),
    # Iteration('Simple | Linear SVC', simple, False, False, lsv()),
    # Iteration('Simple | MLP | Oversampled', simple, False, True, mlp()),
    # Iteration('Simple | Random Forest | Oversampled', simple, False, True, rf()),
    # # Iteration('Simple | Gradient Boosting | Oversampled', simple, False, True, gb()),
    # Iteration('Simple | SGD | Oversampled', simple, False, True, sgd()),
    # Iteration('Simple | Linear SVC | Oversampled', simple, False, True, lsv()),
    # Iteration('Simple | MLP | Scaled', simple, True, False, mlp()),
    # Iteration('Simple | Random Forest | Scaled', simple, True, False, rf()),
    # # Iteration('Simple | Gradient Boosting | Scaled', simple, True, False, gb()),
    # Iteration('Simple | SGD | Scaled', simple, True, False, sgd()),
    # Iteration('Simple | Linear SVC | Scaled', simple, True, False, lsv()),
    # Iteration('Simple | MLP | Scaled | Oversampled', simple, True, True, mlp()),
    # Iteration('Simple | Random Forest | Scaled | Oversampled', simple, True, True, rf()),
    # # Iteration('Simple | Gradient Boosting | Scaled | Oversampled', simple, True, True, gb()),
    # Iteration('Simple | SGD | Scaled | Oversampled', simple, True, True, sgd()),
    # Iteration('Simple | Linear SVC | Scaled | Oversampled', simple, True, True, lsv()),
    #
    # Iteration('Simple Downgrade | MLP', simple_downgrade, False, False, mlp()),
    # Iteration('Simple Downgrade | Random Forest', simple_downgrade, False, False, rf()),
    # Iteration('Simple Downgrade | Gradient Boosting', simple_downgrade, False, False, gb()),
    # Iteration('Simple Downgrade | SGD', simple_downgrade, False, False, sgd()),
    # Iteration('Simple Downgrade | Linear SVC', simple_downgrade, False, False, lsv()),
    # Iteration('Simple Downgrade | MLP | Oversampled', simple_downgrade, False, True, mlp()),
    # Iteration('Simple Downgrade | Random Forest | Oversampled', simple_downgrade, False, True, rf()),
    # # Iteration('Simple Downgrade | Gradient Boosting | Oversampled', simple_downgrade, False, True, gb()),
    # Iteration('Simple Downgrade | SGD | Oversampled', simple_downgrade, False, True, sgd()),
    # Iteration('Simple Downgrade | Linear SVC | Oversampled', simple_downgrade, False, True, lsv()),
    # Iteration('Simple Downgrade | MLP | Scaled', simple_downgrade, True, False, mlp()),
    # Iteration('Simple Downgrade | Random Forest | Scaled', simple_downgrade, True, False, rf()),
    # # Iteration('Simple Downgrade | Gradient Boosting | Scaled', simple_downgrade, True, False, gb()),
    # Iteration('Simple Downgrade | SGD | Scaled', simple_downgrade, True, False, sgd()),
    # Iteration('Simple Downgrade | Linear SVC | Scaled', simple_downgrade, True, False, lsv()),
    # Iteration('Simple Downgrade | MLP | Scaled | Oversampled', simple_downgrade, True, True, mlp()),
    # Iteration('Simple Downgrade | Random Forest | Scaled | Oversampled', simple_downgrade, True, True, rf()),
    # # Iteration('Simple Downgrade | Gradient Boosting | Scaled | Oversampled', simple_downgrade, True, True, gb()),
    # Iteration('Simple Downgrade | SGD | Scaled | Oversampled', simple_downgrade, True, True, sgd()),
    # Iteration('Simple Downgrade | Linear SVC | Scaled | Oversampled', simple_downgrade, True, True, lsv()),
    #
    # Iteration('Simple Downgrade Postpaid | MLP', simple_downgrade_postpaid, False, False, mlp()),
    # Iteration('Simple Downgrade Postpaid | Random Forest', simple_downgrade_postpaid, False, False, rf()),
    # Iteration('Simple Downgrade Postpaid | Gradient Boosting', simple_downgrade_postpaid, False, False, gb()),
    # Iteration('Simple Downgrade Postpaid | SGD', simple_downgrade_postpaid, False, False, sgd()),
    # Iteration('Simple Downgrade Postpaid | Linear SVC', simple_downgrade_postpaid, False, False, lsv()),
    # Iteration('Simple Downgrade Postpaid | MLP | Oversampled', simple_downgrade_postpaid, False, True, mlp()),
    # Iteration('Simple Downgrade Postpaid | Random Forest | Oversampled', simple_downgrade_postpaid, False, True, rf()),
    # # Iteration('Simple Downgrade Postpaid | Gradient Boosting | Oversampled', simple_downgrade_postpaid, False, True, gb()),
    # Iteration('Simple Downgrade Postpaid | SGD | Oversampled', simple_downgrade_postpaid, False, True, sgd()),
    # Iteration('Simple Downgrade Postpaid | Linear SVC | Oversampled', simple_downgrade_postpaid, False, True, lsv()),
    # Iteration('Simple Downgrade Postpaid | MLP | Scaled', simple_downgrade_postpaid, True, False, mlp()),
    # Iteration('Simple Downgrade Postpaid | Random Forest | Scaled', simple_downgrade_postpaid, True, False, rf()),
    # # Iteration('Simple Downgrade Postpaid | Gradient Boosting | Scaled', simple_downgrade_postpaid, True, False, gb()),
    # Iteration('Simple Downgrade Postpaid | SGD | Scaled', simple_downgrade_postpaid, True, False, sgd()),
    # Iteration('Simple Downgrade Postpaid | Linear SVC | Scaled', simple_downgrade_postpaid, True, False, lsv()),
    # Iteration('Simple Downgrade Postpaid | MLP | Scaled | Oversampled', simple_downgrade_postpaid, True, True, mlp()),
    # Iteration('Simple Downgrade Postpaid | Random Forest | Scaled | Oversampled', simple_downgrade_postpaid, True, True, rf()),
    # # Iteration('Simple Downgrade Postpaid | Gradient Boosting | Scaled | Oversampled', simple_downgrade_postpaid, True, True, gb()),
    Iteration('Simple Downgrade Postpaid | SGD | Scaled | Oversampled', simple_downgrade_postpaid, True, True, sgd()),
    # Iteration('Simple Downgrade Postpaid | Linear SVC | Scaled | Oversampled', simple_downgrade_postpaid, True, True, lsv()),
]

# execute each iteration and show its precision and recall

for iteration in iterations:
    print("\nITERATION: {}".format(iteration.name))
    training_features, training_labels, test_features, test_labels =\
        iteration.datasets.get(iteration.scaled, iteration.oversampled)
    iteration.classifier.fit(training_features, training_labels)
    actual = test_labels
    predictions = iteration.classifier.predict(test_features)
    # debug: catch cases where some classifiers never predict particular values
    missing_predictions = set(actual) - set(predictions)
    print("missing predictions: {}".format(missing_predictions))
    # print("CONFUSION MATRIX: \n{}".format(
    #     confusion_matrix(actual, predictions, labels = [0, 1])
    # ))
    print("PRECISION: {:.2f}".format(
        precision_score(actual, predictions, labels = iteration.datasets.measured_labels, average = 'weighted')
    ))
    print("RECALL: {:.2f}".format(
        recall_score(actual, predictions, labels = iteration.datasets.measured_labels, average = 'weighted')
    ))
