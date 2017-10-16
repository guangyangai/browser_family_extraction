# **Report for agent family prediction**


---

**Preidct agent family (and version) based on agent description**

The goals / steps of this project are the following:
1. Predict family and version number of an agent based on its description
2. Summarize findings and thoughts during such process


---

### Reflection

### 1. Summary
First of all, there are 2 tasks for this project:
1. predicting agent family, which is a classification problem
2. predicting version number which is also a classification problem. We should think the version number as a string/categorical variable, instead of a continuous int variable.
Given the size(421215 records which could be easily loaded into memory) and nature of the data, I decided to use batch, model-based learning.

![alt text][family]
![alt text][version]

[family]: https://github.com/energydatasci/browser_family_extraction/blob/master/images/family_countplot.png "Family Distribution in the Data"
[version]: https://github.com/energydatasci/browser_family_extraction/blob/master/images/version_countplot.png "version Distribution in the Data"


The pipeline of the project is:
1. Clean data using pandas dataframe functions (e.g. describe and info). More specifically, I checked the type, unique values and their counts for each column and whether there
are missing value in any row. I dropped the rows where there are missing value (e.g. None for version number) for the training. I also dropped the rows where the predicted values for thoses row only had a few occurrences which
are not large enough for training.
2. Split the training data into test set and training set using stratified sampling, due to the fact that the agent family in the training data are not uniformly distributed. (Add figure here)
3. Extract features from agent description
4. Train a classifier on the training set for agent family and version number (Use GridSearch or Randomized Search to fine tune the classifier)
5. Evaluate the performance on the test set
6. If the performance is good enough, use the trained classifier to predict on the test data


### 2. Text feature extraction on the agent description

I decided to use  `CountVectorizer` in sklearn to create features.

The first tool coming into my mind is **bag of words** for the description, namely:

1.tokenizing strings and giving an integer id for each possible token,  by using white-spaces as token separators.
2.counting the occurrences of tokens in each agent description.
3.normalizing and weighting with diminishing importance tokens that occur in the majority of agent descriptions.

In this scheme, features and samples are defined as follows:
each individual token occurrence frequency is treated as a feature, and the vector of all the token frequencies for a given agent is considered a multivariate sample.

A collection of agents can thus be represented by a matrix with one row per agent and one column per token (e.g. word) occurring in the collection.

I used 2-ngram to generate the tokens, because by string(e.g. applewebkit) itself it could be meaningless and not correlated to agent family.
But string plus version number or IP address is more meaningful. Further I used TF-IDF weighting to reduce the weights of longer agent description and common word combinations.

### 3. Classifier choice
For a reasonable amount of labeled data (in our case 336972 labeled training set), SVM seems to be good choice. Out of the sklearn classes
for SVM classification,`SGDClassifier` is a good choice as it has the least time complexity (O(m*n)) and has out-of-core support.


### 4. Possible improvements to your pipeline
####4.1 Use distributed learning or online learning to scale to larger data set

Because the sample size is small, I loaded all the training data in memory for the training. However, in order to make the pipeline scalable to
much larger sized data, I could build a online learning or distributed learning pipeline.
####4.2 Use ensemble model

To deploy a linear SVM classifier, it's better to design an application that overlays a Boolean rule-based classifier (e.g. decision trees) over the machine learning classifier.
Users frequently like to adjust things that do not come out quite right, and if management gets on the phone and wants the classification of a particular document fixed,
then this is much easier to do by hand-writing a rule than by working out how to adjust the weights of an SVM without destroying the overall classification accuracy.



