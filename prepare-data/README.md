# Description
Contains the various datasets used by this prototype.

- __all-labelled.csv:__
Original dataset (see below) labelled using [this Groovy code](https://github.com/Peter-Martin/mobile-subscribers/blob/master/prepare-data/src/Label.groovy).

- __all-original.csv:__
Original full dataset created by [Generate Data](http://generatedata.com/), then cleaned using [this Groovy code](https://github.com/Peter-Martin/mobile-subscribers/blob/master/prepare-data/src/Clean.groovy).

- __test-labelled.csv:__
Subset (20%) of the labelled dataset above, used for testing model predictions.

- __training-labelled.csv:__
Subset (60%) of the labelled dataset above, used for training the model.

- __validation-labelled.csv:__
Subset (20%) of the labelled dataset above, optionally used for validating the trained model.

*Note: See [this article](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7) for an
explanation of the purpose of the above training, validation and test datasets.*
