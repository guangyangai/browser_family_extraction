# **Report for agent family prediction**


---

**Preidct agent family (and version) based on agent description**

The goals / steps of this project are the following:
1. Predict family of an agent based on its description
2. Summarize findings and thoughts during such process


---

### Reflection

### 1. Summary
First of all, there are 2 tasks for this project:
1. predicting agent family, which is a classification problem
2. predicting version number which is a regression problem.
Given the size(421215 records which could be easily loaded into memory) and nature of the data, I decided to use batch, model-based learning.

![alt text][family]
![alt text][version]

[family]: https://github.com/energydatasci/browser_family_extraction/blob/master/images/family_countplot.png "Family Distribution in the Data"
[version]: https://github.com/energydatasci/browser_family_extraction/blob/master/images/version_countplot.png "version Distribution in the Data"


The pipeline of the project is:
1. Split the training data into test set and training set using stratified sampling, due to the fact that the agent family in the training data are not uniformly distributed. (Add figure here)
2. Gain insights on the data using pandas dataframe functions (e.g. describe and info). More specifically, I checked the type and uniqueness of each column and whether there
are missing values in any row. I need to drop the row where there are missing values for the training.
3. Extract features from the agent description
4. Train a classifier on the training set for agent family and train a regressor for version number
5. Validate the performance on the test set
6. Use the trained classifier to predict on the test data


### 2. Text feature extraction on the agent description

I did a simple desribe on the data and noticed all the agent descriptions are unique, apparentlly I can not use the raw agent description
to train the model, it would need some pre-processing on its description to exact some features to train the model.
I decided to use text feature extraction modules in sklearn and applied on agent description to create features.

The first tool coming into my mind is bag of words for the description, namely:

1.tokenizing strings and giving an integer id for each possible token,  by using white-spaces and brackets as token separators.
2.counting the occurrences of tokens in each agent description.
3.normalizing and weighting with diminishing importance tokens that occur in the majority of agent descriptions.

In this scheme, features and samples are defined as follows:
each individual token occurrence frequency (normalized or not) is treated as a feature.
the vector of all the token frequencies for a given agent is considered a multivariate sample.

A collection of agents can thus be represented by a matrix with one row per agent and one column per token (e.g. word) occurring in the collection.

By closer looking into the agent, it's actually made of a series of patterned strings separated by white space. The patten of the token is
'\W+ / number (\W)'
default vectorizer with 1 ngram
<421215x33719 sparse matrix of type '<class 'numpy.int64'>', 33719 words is too large.

Performance comparison for SVM classification:



### 3. Suggest possible improvements to your pipeline

Instead of building a simple collection of unigrams (n=1), I built a collection of bigrams (n=2), where occurrences of pairs of consecutive words are counted.

with 2 ngrams
<421215x152044 sparse matrix of type '<class 'numpy.int64'>'
	with 7022630 stored elements in Compressed Sparse Row format>
2 ngram make sense because by platform itself could be meaningless, but with version number or IP address is meaningful.

I used TF-IDF weighting to reduce the weights of longer agent description and common word combinations.

### 3. Classifier choice
If you have fairly little data and you are going to train a supervised classifier, the machine learning theory says you should stick to a
classifier with high bias. There are theoretical and empirical results that Naive Bayes does well in such circumstances.
Another choice is to try semi-supervised training model.

If there is a reasonable amount of labeled data (in our case 336972 labeled training set), then you are in the perfect position to use everything that we have presented about text classification.
For instance, you may wish to use an SVM. However, if you are deploying a linear classifier such as an SVM, you should probably design an application that overlays a Boolean rule-based classifier over the machine learning classifier.
Users frequently like to adjust things that do not come out quite right, and if management gets on the phone and wants the classification of a particular document fixed right now,
then this is much easier to do by hand-writing a rule than by working out how to adjust the weights of an SVM without destroying the overall classification accuracy.
This is one reason why machine learning models like decision trees which produce user-interpretable Boolean-like models retain considerable popularity.

Our choice could be decision tree + SVM.
