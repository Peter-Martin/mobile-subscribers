# =============================================================================
# DNNClassifier for Mobile Subscriber dataset.
#
# Uses HIGH-LEVEL TensorFlow APIs -> Estimators
#   + some mid-level APIs         -> Datasets
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import subscriber_data

parser = argparse.ArgumentParser()
#parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--batch_size', default=1000, type=int, help='batch size')
#parser.add_argument('--train_steps', default=1000, type=int,
#parser.add_argument('--train_steps', default=10000, type=int,
parser.add_argument('--train_steps', default=10000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # -------------------------------------------------------------------------
    # 1) Fetch the data

    (train_x, train_y), (validation_x, validation_y), (test_x, test_y) = subscriber_data.load_data()

    # Feature columns describe how to use the input.
    # numeric means use directly; not weighted, etc.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # -------------------------------------------------------------------------
    # 2) Build 2 hidden layer DNN with 10, 10 units respectively.

    # note that defaults are:
    #   RELU activation_fn
    #   Adagrad optimizer (training algorithm)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 5 classes.
        n_classes=5)

    # -------------------------------------------------------------------------
    # 3) Train the Model

    classifier.train(
        input_fn=lambda:subscriber_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # -------------------------------------------------------------------------
    # 4) Evaluate the model using cross-validation set

    eval_result = classifier.evaluate(
        input_fn=lambda:subscriber_data.eval_input_fn(validation_x, validation_y,
                                                args.batch_size))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # -------------------------------------------------------------------------
    # 5) Make predictions for test set

    predictions = classifier.predict(
        input_fn=lambda:subscriber_data.eval_input_fn(test_x,
                                                      labels=None,
                                                      batch_size=args.batch_size))

    # -------------------------------------------------------------------------
    # 6) Show some rough statistics for prediction results

    num_changed = 0
    num_changed_correct_value = 0
    num_changed_incorrect_value = 0

    for pred_dict, expected in zip(predictions, test_y):
        if expected != 3:
            num_changed += 1
        predicted = pred_dict['class_ids'][0]
        if predicted != 3:
            if predicted == expected:
                num_changed_correct_value += 1
            else: # spotted a change, but didn't spot correct change
                num_changed_incorrect_value += 1
    print("\nNum changed is {}, Num changed correct is {}, Num changed incorrect is {}, "
          "Percentage changed correct is {:.1f}".
          format(num_changed, num_changed_correct_value, num_changed_incorrect_value,
                 100 * (num_changed_correct_value / num_changed)))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
