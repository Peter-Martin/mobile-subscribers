# Description
Trains machine learning models using [scikit-learn](http://scikit-learn.org/stable/index.html).

__auto-sklearn.py__

Automatically trains a model using scikit-learn and auto-sklearn.
This produces a less successful model than the iterative training below.

__iterate.py__

Iteratively trains model combinations using scikit-learn,
i.e., by iteratively varying the classifier,
dataset (full/simplified/postpaid-only), scaling (true/false)
and oversampling (true/false). This file has the main
training activity for this prototype.

__manual-sklearn.py__

Manually trains models using scikit-learn.
This file is useful for quickly trying out new or different
model combinations.

# Prerequisites

[Install scikit-learn](http://scikit-learn.org/stable/install.html)

[Install auto-sklearn](https://automl.github.io/auto-sklearn/stable/installation.html)
