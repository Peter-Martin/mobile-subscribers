# Overview

For any business that sells services to consumers, two key forecasting
questions are:

  - Which consumers are likely to **cancel or downgrade** their service?
  - Which consumers are likely to **upgrade** their service?

For the former (the potential ‘downgraders’), the business may wish to
preemptively avoid the scenario by offering an alternative service
package that’s more suited to the consumer’s needs.

For the latter (the potential ’upgraders’), the business may wish to
maximise consumer revenue by preemptively selling an upgraded service
package or even upselling additional services.

So it’s clear that early identification of potential
downgraders/upgraders can offer significant business benefits, both in
terms of reducing consumer ‘churn’ and maximising business revenue.

For mobile phone operators, it’s typically very simple for a subscriber
to downgrade their phone plan or even to cancel and switch to another
operator. Consequently, downgrades and churn are significant issues.

On the other hand, mobile phone operators typically offer a wide range
of plans and services, so maximising upgrade and upsell of these
plans/services can offer significant revenue benefits.

Machine Learning potentially offers a means of using subscriber data to
predict who will downgrade/cancel or upgrade. In Machine Learning
terminology, this is a classification task.

For mobile phone operators, however, stringent data privacy controls and
physical dispersal of data (across multiple subsystems or even
locations) often means that a comprehensive, ‘wide’ set of subscriber
data is not readily available for machine learning classification.

Thus, the rest of this write-up describes a prototype that seeks to
predict downgrades/upgrades from a very reduced (‘narrow’) set of
subscriber data; i.e., with key items (features):

  - prepaid or postpaid
  - contract length (for postpaid)
  - subscriber spend
  - number of calls per month
  - purchased versus used calls/texts/data per month

The above data is quite anonymous (thus less privacy-sensitive) and
should be quite readily available, e.g., from the mobile phone
operator’s billing system and associated subsystems.

The prototype seeks to leverage the fact that a lot of important
information relevant to downgrade/upgrade prediction can be **inferred**
from the aforementioned data items; for example:

  - Is the subscriber ‘steady’ (long postpaid contract) or more ‘fickle’
    (short postpaid contract or prepaid)?
  - Is the subscriber quite spendthrift (high spend and purchases much
    more calls/texts/data than they actually use) or a more careful
    spender (low spend and only purchases what they need)?
  - Is the subscriber using almost the maximum of their purchased levels
    of calls/text/data? This may suggest they’re likely to upgrade.
  - What age is the subscriber? This can be roughly inferred from usage
    patterns; e.g., younger subscribers tend to use more data and less
    calls/texts than older subscribers. Younger subscribers may also be
    more ‘fickle’; i.e., more likely to switch operators.
  - Is the subscriber a business, thus more likely to regularly
    review/audit their spend? This can be roughly inferred from a high
    number of short calls and high data usage.

# Data Preparation

For this prototype, a *synthetic* subscriber data set was created using
the following steps:

**1)** Generate the initial random data set using
[generate data](http://generatedata.com/).
The script was run locally to avail of an
increased data set limit of 100,000 items. Generation was run 10 times
to produce a data set with a total of 1,000,000 subscribers. This data
set is available [here](https://github.com/Peter-Martin/mobile-subscribers/blob/master/prepare-data/all-original.csv.tar.gz).

**2)** Clean the initial data set to fix and/or remove any invalid
values. A [custom Groovy script](https://github.com/Peter-Martin/mobile-subscribers/blob/master/prepare-data/src/Clean.groovy)
was written to perform the cleaning.

**3)** Label the data set; i.e., classify each subscriber as
cancelled/downgraded/unchanged/upgraded according to the
rules/inferences listed towards the end of the earlier Overview
section. Note, however, that an element of
real-world randomness was also included; e.g., so that a small number of
apparently more ‘stable’ subscribers still downgrade or cancel, etc.

An [additional Groovy script](https://github.com/Peter-Martin/mobile-subscribers/blob/master/prepare-data/src/Label.groovy)
was written to perform this labelling.

**4)** Split the labelled dataset into training (60%), validation (20%)
and test (20%) sets. See
[this article](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)
for an explanation of the purpose of each of these sets.

The original full labelled dataset is available
[here](https://github.com/Peter-Martin/mobile-subscribers/blob/master/prepare-data/all-labelled.csv.tar.gz).
The set of original labels and their values is as follows:

<table align="left">
<tbody>
<tr class="odd">
<td><p><b>UpdatedIn90Days</b></p>
<p>The subscriber’s action within 90 days of the data sampling:</p>
<table>
<tr><td>- upgraded plan</td><td><i>for prepaid, means switched to postpaid</i></td></tr>
<tr><td>- unchanged</td><td></td></tr>
<tr><td>- downgraded plan</td><td><i>postpaid only</i></td></tr>
<tr><td>- switched to prepaid</td><td><i>postpaid only</i></td></tr>
<tr><td>- cancelled</td><td><i>prepaid stopped topups, postpaid cancelled contract</i></td></tr>
</table>
Note that some of the prototype iterations simplified the above label set; this is described further down.</td>
</tr>
<tr class="even">
<td><p><b>PurchasedAdditionalIn90Days</b></p>
<p>Whether or not the subscriber purchased additional products (from the mobile operator) within 90 days of the data sampling:</p>
<p>
- purchased additional products<br>
- didn't purchase additional products
</p>
<p><strong>NB:</strong> This label was generated in the original dataset. However, in order to narrow its focus, the prototype didn’t attempt to predict this label value.</p></td>
</tr>
</tbody>
</table>

# Software Evaluation

There are many open-source and commercial software tools and packages
available for Machine Learning. Some of the more widely-used packages
were evaluated for this prototype:

[**TensorFlow**](https://www.tensorflow.org/)

Widely-used Python-based machine learning framework. Using TensorFlow’s
full functionality requires quite a strong proficiency in Python. While
bindings for Java (using JNI) and JavaScript are available, these don’t
support all of the functionality of the Python API.

[**scikit-learn**](http://scikit-learn.org/stable)

Also Python-based, but simpler to use than TensorFlow. Consequently,
it’s suitable for use by developers with less proficiency in Python.

[**H2O AI**](https://www.h2o.ai)

Also Python and R-based. One of H2O’s advantages is that it includes a
browser-based UI (‘Flow’) that provides a very convenient way of loading
data and training an initial machine learning model.

TensorFlow was used for the initial machine learning prototyping.
However, based on the above evaluation, scikit-learn was subsequently
used for all of the prototyping.

Note that, for machine-learning production environments, the choice of
software should be re-evaluated. The following factors, among others,
would need to be evaluated for the above software tools/packages:

  - Performance, including usage of GPUs ,etc.
  - Cloud availability.
  - Clustering capability.
  - Integration with pipeline (e.g., Apache Spark) that feeds data to
    the machine learning model.

# Model Training and Prediction

A number of different iterations were performed during model training.
Each iteration trained a new model by incrementally varying the
classifiers and/or applying various transformations to the input data:

## Classifiers

The following classifiers were used during the training iterations:

  - [**Multi-layer Perceptron Classifier**](http://scikit-learn.org/stable/modules/neural_networks_supervised.html)
  based on neural networks.
  - [**Random Forest Classifier**](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  based on decision trees.
  - [**Gradient Boosting Classifier**](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
  based on an ensemble of decision trees.
  - [**Stochastic Gradient Descent Classifier**](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
  based on iteratively reducing the loss/error/cost associated with a model.
  - [**Linear Support Vector Classifier**](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
  based on categorising examples.

Note that each training iteration used an identically-configured
classifier (same random state, same number of hidden layers for neural
network, etc.). It may be possible to obtain improved prediction results
by varying the configuration of each classifier..

## Data Sets

The following subscriber datasets were used during the training
iterations:

[**Full**](https://github.com/Peter-Martin/mobile-subscribers/tree/master/prepare-data/one-label)

Includes all of the upgrade and downgrade classes as described earlier.
Includes both postpaid and prepaid subscribers. The full dataset
(training, validation and test) is available at

[**Simple**](https://github.com/Peter-Martin/mobile-subscribers/tree/master/prepare-data/one-label/simple)

Generated by reducing the multiple downgrade classes (described for the
full dataset above) to a single ‘downgrade’ class that represents any of
cancelled, switched to prepaid or the original downgraded label. The
simple downgrade dataset (training, validation and test) is available at

[**Simple Downgrade**](https://github.com/Peter-Martin/mobile-subscribers/tree/master/prepare-data/one-label/simple/downgrade)

Generated by removing the upgrade label from the above simple dataset.
The simple downgrade dataset (training, validation and test) is
available at

[**Simple Downgrade Postpaid**](https://github.com/Peter-Martin/mobile-subscribers/tree/master/prepare-data/one-label/simple/downgrade/postpaid)

Generated by removing the prepaid subscribers from the above simple
downgrade dataset; i.e., so that only postpaid subscribers remain. The
simple downgrade postpaid dataset (training, validation and test) is
available at

## Additional Transformations

During training iterations, the following transformations were applied
to the datasets described above:

  - **Feature Scaling** to normalize feature values; i.e., so that all
    feature values are within a similar range.
  - **Oversampling** (using SMOTE) to balance the relative distribution
    of the various classes; i.e., so that there are fairly similar
    numbers of subscribers that downgrade, upgrade, stay unchanged, etc.

Note that some classifiers are more sensitive than others to the above
transformations.

Also note that validation wasn’t explicitly/externally applied while
performing training iterations; some of the earlier classifiers
internally apply nested k-fold cross-validation.

The following metrics were applied in order to score each iteration’s
predictions:

  - **Precision:** Reflects the prediction ‘quality’ of the model; this
    figure is higher for less false positives.
  - **Recall:** Reflects the prediction ‘quantity’ achieved by the
    model; the figure is higher with more true positives.
  - **F1 Score:** Combines the precision and recall so they can be
    compared across the various training iterations. The F1 score is
    defined as 2(Precision\*Recall) / (Precision+Recall)

In order to compare like-for-like metrics across iterations, the metrics
above were rolled-up to only cover the simple downgrade true/false
scenario. That scenario applies across all of the data sets mentioned
earlier, it’s thus available for comparison all iterations. So, when
taking prediction metrics for the full data set, the
cancelled/switched-to-prepaid/downgraded classes were all simply counted
as ‘downgraded’.

Note that similar metrics could have been measured for predicting
upgrade scenarios; this could be achieved with the following simple
changes:

**i)** Use the ‘Simple Upgrade’ and ‘Simple Upgrade Postpaid’ datasets
instead of their downgrade equivalents.

**ii)** Take prediction metrics (precision and recall) for upgrade
instead of for downgrade.

The Python source code for the training model iterations is available
[here](https://github.com/Peter-Martin/mobile-subscribers/blob/master/train-model/scikit-learn/).

# Downgrade Results and Analysis

The following table describes the training iterations and their
resulting downgrade prediction metrics; the iterations are sorted by their F1
Scores. Note that the table excludes any iterations that didn’t complete
or that failed to predict some classes.

| *Train*                   | *Train*                  | *Train*    | *Train*         | *Predict*     | *Predict*  | *Predict*    |
| :------------------------ | :----------------------- | ---------- | --------------- | ------------- | ---------- | ------------ |
| **Data Set**              | **Classifier**           | **Scaled** | **Oversampled** | **Precision** | **Recall** | **F1 Score** |
| Simple Downgrade Postpaid | MLP                      | yes        | yes             | 0.47          | 0.58       | 0.52         |
| Simple Downgrade Postpaid | SGD                      | yes        | yes             | 0.43          | 0.61       | 0.50         |
| Simple Downgrade Postpaid | Linear SVC               | yes        | yes             | 0.41          | 0.58       | 0.48         |
| Simple Downgrade          | MLP                      | yes        | yes             | 0.35          | 0.73       | 0.47         |
| Simple Downgrade Postpaid | Random Forest            | yes        | yes             | 0.42          | 0.54       | 0.47         |
| Simple                    | MLP                      | yes        | yes             | 0.35          | 0.72       | 0.47         |
| Simple Downgrade          | MLP                      | no         | yes             | 0.36          | 0.68       | 0.47         |
| Simple Downgrade Postpaid | Gradient Boosting        | no         | no              | 0.62          | 0.37       | 0.46         |
| Simple Downgrade Postpaid | MLP                      | yes        | no              | 0.63          | 0.36       | 0.46         |
| Simple Downgrade Postpaid | Linear SVC               | no         | yes             | 0.32          | 0.77       | 0.45         |
| Simple Downgrade Postpaid | Random Forest            | no         | yes             | 0.46          | 0.43       | 0.44         |
| Simple Downgrade Postpaid | MLP                      | no         | yes             | 0.30          | 0.85       | 0.44         |
| Simple Downgrade Postpaid | Random Forest            | no         | no              | 0.49          | 0.40       | 0.44         |
| Simple Downgrade Postpaid | Random Forest            | yes        | no              | 0.49          | 0.40       | 0.44         |
| Simple                    | SGD                      | yes        | yes             | 0.33          | 0.62       | 0.43         |
| Simple                    | Linear SVC               | yes        | yes             | 0.31          | 0.69       | 0.43         |
| Simple Downgrade          | Random Forest            | yes        | yes             | 0.35          | 0.51       | 0.42         |
| Simple                    | Random Forest            | yes        | yes             | 0.36          | 0.49       | 0.42         |
| Simple                    | MLP                      | no         | yes             | 0.29          | 0.70       | 0.41         |
| Simple Downgrade          | SGD                      | no         | yes             | 0.30          | 0.62       | 0.40         |
| Simple Downgrade          | SGD                      | yes        | yes             | 0.27          | 0.79       | 0.40         |
| Simple Downgrade          | Linear SVC               | yes        | yes             | 0.27          | 0.79       | 0.40         |
| Simple Downgrade          | Random Forest            | no         | yes             | 0.42          | 0.38       | 0.40         |
| Simple                    | Random Forest            | no         | yes             | 0.42          | 0.37       | 0.39         |
| Simple Downgrade          | Gradient Boosting        | no         | no              | 0.63          | 0.28       | 0.39         |
| Simple Downgrade          | MLP                      | yes        | no              | 0.63          | 0.28       | 0.39         |
| Simple Downgrade          | Random Forest            | no         | no              | 0.48          | 0.32       | 0.38         |
| Simple Downgrade          | Random Forest            | yes        | no              | 0.48          | 0.32       | 0.38         |
| Simple                    | Random Forest            | no         | no              | 0.47          | 0.32       | 0.38         |
| Simple                    | Random Forest            | yes        | no              | 0.47          | 0.32       | 0.38         |
| Simple                    | Gradient Boosting        | no         | no              | 0.63          | 0.26       | 0.37         |
| Simple Downgrade          | Linear SVC               | no         | yes             | 0.23          | 0.88       | 0.36         |
| Simple                    | MLP                      | yes        | no              | 0.62          | 0.25       | 0.36         |
| Simple                    | Linear SVC               | no         | yes             | 0.20          | 0.81       | 0.32         |
| Simple Downgrade          | SGD                      | no         | no              | 0.19          | 0.93       | 0.32         |
| Simple                    | Linear SVC               | no         | no              | 0.18          | 0.94       | 0.30         |
| Full                      | MLP                      | yes        | yes             | 0.22          | 0.43       | 0.29         |
| Simple                    | SGD                      | no         | yes             | 0.26          | 0.29       | 0.27         |
| Full                      | Random Forest            | yes        | yes             | 0.20          | 0.32       | 0.25         |
| Full                      | Random Forest            | no         | yes             | 0.25          | 0.23       | 0.24         |
| Simple Downgrade Postpaid | Linear SVC               | no         | no              | 0.25          | 0.19       | 0.22         |
| Full                      | Random Forest            | no         | no              | 0.31          | 0.16       | 0.21         |
| Full                      | Random Forest            | yes        | no              | 0.31          | 0.16       | 0.21         |
| Full                      | Linear SVC               | yes        | yes             | 0.12          | 0.46       | 0.19         |
| Full                      | SGD                      | yes        | yes             | 0.14          | 0.29       | 0.19         |
| Full                      | MLP                      | no         | yes             | 0.14          | 0.17       | 0.15         |
| Simple Downgrade Postpaid | SGD                      | no         | yes             | 0.20          | 0.10       | 0.13         |
| Simple Downgrade          | MLP                      | no         | no              | 0.69          | 0.05       | 0.09         |
| Full                      | SGD                      | no         | yes             | 0.09          | 0.08       | 0.08         |
| Simple Downgrade          | Linear SVC               | no         | no              | 0.26          | 0.05       | 0.08         |
| Full                      | Linear SVC               | no         | yes             | 0.05          | 0.08       | 0.06         |
| Simple Downgrade Postpaid | SGD                      | no         | no              | 0.21          | 0.01       | 0.02         |
| Simple Downgrade Postpaid | MLP                      | no         | no              | 0.17          | 0.00       | 0.00         |
| Simple                    | SGD                      | no         | no              | 0.15          | 0.00       | 0.00         |
| Full                      | Linear SVC               | no         | no              | 0.13          | 0.00       | 0.00         |

The confusion matrices for the top 2 performers (by F1 score) are as
follows; the matrices respectively show the metrics for *downgraded* and
*unchanged* classes.

<p></p>

|--- HIGHEST F1 SCORE ---<br/><br/>- Simple Downgrade Postpaid Dataset<br/>- Multi-layer Perceptron Classifier<br/>- Scaled<br/>- Oversampled|
|:---|
|*\[\[15069 11039\]*<br/>*\[16726 55680\]\]*|
|For predicting downgrades, this model iteration produced **15,069 true positives**; i.e., subscribers successfully predicted as downgrades. However, the model failed to predict 11,039 downgrades, thus it has a recall (‘quantity’) score of 0.58.|
|In addition, the model produced **16,726 false positives**; i.e., erroneously predicted subscribers as downgraded when they were actually unchanged. This is from the total unchanged figure of (16,726 + 55,680). Thus, the model has a reasonable precision (‘quality’) score of 0.47.|


|--- 2ND HIGHEST F1 SCORE ---<br/><br/>- Simple Downgrade Postpaid Dataset<br/>- Stochastic Gradient Descent Classifier<br/>- Scaled<br/>- Oversampled|
|:---|
|*\[\[16069 10039\]*<br/>*\[21660 50746\]\]*|
|This model iteration scored similarly to the previous model; slightly better recall (‘quantity’) score of 0.61, but a slightly lower precision (‘quality’) score of 0.43.|

It’s also useful to examine a very different confusion matrix for
another model iteration:

|--- LOWER F1 SCORE ---<br/><br/>- Simple Downgrade Postpaid Dataset<br/>- Multi-layer Perceptron Classifier<br/>- Not Scaled<br/>- Oversampled|
|:---|
|*\[\[22156 3952\]*<br/>*\[51695 20711\]\]*|
|For predicting downgrades, this model iteration produced **22,156 true positives**; i.e., subscribers successfully predicted as downgrades. The model only failed to predict 3,952 downgraded subscribers. So it has a high recall (‘quantity’) score of 0.85.|
|However, the model produced **51,695 false positives**; i.e., erroneously predicted subscribers as downgraded when they were actually unchanged. This is from the total unchanged figure of (51,695 + 20,711). So the model has a low precision (‘quality’) score of 0.30.|

For the above model iterations, the precision and recall could be
fine-tuned by adjusting the *probability threshold* (aka *decision
threshold*); i.e., the point at which a subscriber is adjudged to be
either downgraded or unchanged. This is described for scikit-learn in
[this article](https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65).

An additional tuning measure would be to use the validation datasets
(created at the same time as the training datasets earlier) to
explicitly perform validation. This could be used to identify any
potential *underfitting (high bias)* or *overfitting (high variance)* by
the various model iterations:

  - **Underfitting** is represented by a high error rate in both the training and
    validation sets. This may be addressed by increasing the number of
    subscriber features in the datasets; e.g., age, gender, etc.
  - **Overfitting** is represented by a low error rate in the training set, but
    a high error rate in the validation set. This may be addressed by reducing the
    number of subscriber features or by obtaining data from a greater
    number of subscribers.

Applying non-nested (cross-) validation is described for scikit-learn
[here](http://scikit-learn.org/stable/modules/cross_validation.html).

**Note:** Automatic machine learning (using scikit-learn) was also
evaluated during the course of this prototype. However, it produced a
less successful model (lower precision and recall) than the models
produced iteratively above. The source code for the automatic machine
learning is available
[here](https://github.com/Peter-Martin/mobile-subscribers/blob/master/train-model/scikit-learn/auto-sklearn.py).

# Conclusions

**1\) Simple and Specific**

  - Employing scaling/over-sampling typically delivered small
    incremental metric improvements; e.g., of a few percent.
  - Greater improvements were observed by manipulating/engineering the
    data; e.g., separating unrelated data (postpaid from prepaid),
    reducing the number of classes, etc.
  - The best results were achieved when the number of classes was
    simplified to downgraded/unchanged/upgraded and when postpaid
    subscribers were separated from prepaid; i.e., when there was less
    ‘noise’ in the input data.

This conclusion may be stated as follows:

For machine learning classification, it's best to ask a *simple* question,
then provide very *specific* data to allow that question to be answered.

**2\) Improving Prediction Rates**

As-is, the downgrade prediction metrics shown in the earlier table
wouldn’t be sufficient for a real-life production environment. However,
the following steps could be taken to improve the prediction rate: 

  - Use real data; the synthetic data (generated specifically for this
    prototype) applies its own rules/inferences, so it has some inbuilt
    biases.
  - Include additional subscriber features; e.g., age, gender,
    personal/business, customer loyalty (period of time with the
    operator), etc.
  - Include comprehensive
  [feature engineering](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/).
  - Include mobile operator metrics/features captured at the time of
    subscriber downgrade/upgrade; average customer review, customer care
    responsiveness, etc.

Overall, the greater ‘return’ for effort was found from manipulating and
improving the input data, rather than adjusting the classifier
algorithms and parameters. This is where most future efforts should be
concentrated.

This confirms what Andrew Ng states in his course notes for
[Stanford University Machine Learning](https://www.coursera.org/learn/machine-learning):

*“It’s not who has the best algorithm that wins. It’s who has the most
data.”*

(Banko and Brill, 2001)
