# Description
Trains a machine learning model using [TensorFlow](https://www.tensorflow.org/)

>**NB:** **This model doesn't produce satisfactory results**;
the model likely has coding and/or configuration errors.
Please instead use the scikit-learn training code located
[here](https://github.com/Peter-Martin/mobile-subscribers/tree/master/train-model/scikit-learn).

__subscriber_data.py__

Loads the mobile subscriber dataset.

__subscriber_estimator.py__

Trains a TensorFlow DNNClassifier (deep neural network classifier)
for the mobile subscriber dataset.

# Prerequisites

[Install TensorFlow](https://www.tensorflow.org/install/)
