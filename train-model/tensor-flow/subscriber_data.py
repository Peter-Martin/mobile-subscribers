# =============================================================================
# Load Mobile Subscriber dataset for TensorFlow Estimator.
# =============================================================================

import pandas as pd
import tensorflow as tf

TRAIN_URL = "file:///home/peter/Documents/projects/MobileSubscribers/training-labelled.csv"
VALIDATION_URL = "file:///home/peter/Documents/projects/MobileSubscribers/validation-labelled.csv"
TEST_URL = "file:///home/peter/Documents/projects/MobileSubscribers/test-labelled.csv"

CSV_COLUMN_NAMES = [
    'Postpaid', 'ContractLengthWeeks', 'AveragePayment', 'NumberOfPayments', 'AveragePeakCreditBalance',
    'VoicePurchasedMinutesPerMonth', 'VoiceUsedMinutesPerMonth', 'VoiceNumberOfCallsPerMonth',
    'SmsPurchasedPerMonth', 'SmsUsedPerMonth', 'DataMbPurchasedPerMonth', 'DataMbUsedPerMonth',
    'UpdatedIn90Days', 'PurchasedAdditionalIn90Days'
]
UPDATED_IN_90_DAYS = ['CANCELLED','SWITCHED_TO_PREPAID','DOWNGRADED','UNCHANGED','UPGRADED']

def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    validation_path = tf.keras.utils.get_file(VALIDATION_URL.split('/')[-1], VALIDATION_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    return train_path, validation_path, test_path

def load_data(y1_name='UpdatedIn90Days', y2_name='PurchasedAdditionalIn90Days'):
    # Returns the subscriber dataset as (train_x, train_y), (test_x, test_y).
    train_path, validation_path, test_path = maybe_download()
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=1)
    # train now holds a pandas DataFrame, which is data structure
    # like a table; this is key'ed on CSV column names
    # 1. Assign the DataFrame's labels (the 2 right-most columns) to train_y1/y2,
    #    note that we delete (pop) the labels from the DataFrame.
    # 2. Assign the remainder of the DataFrame to train_x
    train_x, train_y1, train_y2 = train, train.pop(y1_name), train.pop(y2_name)
    # similar to above for validation and test sets
    validation = pd.read_csv(validation_path, names=CSV_COLUMN_NAMES, header=1)
    validation_x, validation_y1, validation_y2 = validation, validation.pop(y1_name), validation.pop(y2_name)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=1)
    test_x, test_y1, test_y2 = test, test.pop(y1_name), test.pop(y2_name)
    # Return six pandas DataFrames
    # NB: currently only training for y1 (UpdatedIn90Days)
    return (train_x, train_y1), (validation_x, validation_y1), (test_x, test_y1)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    # Return the dataset.
    return dataset

# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    # Separate the label from the features
    #label = features.pop('Species')
    # pop off last label, unused for now
    features.pop('PurchasedAdditionalIn90Days')
    # 2nd last label is one we're using for now
    label = features.pop('UpdatedIn90Days')
    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)
    # Parse each line.
    dataset = dataset.map(_parse_line)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the dataset.
    return dataset
