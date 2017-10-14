# **Report for agent family prediction**


---

**Preidct agent family (and version) based on agent description**

The goals / steps of this project are the following:
* Predict family of an agent based on its description
* Summarize findings and thoughts during such process


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Summary
First of all, predicting agent family is a classification problem while predicting version number is a regression problem.
Given the size(421215 rows) and nature of the data, I decided to use batch, model-based learning.



Needs to check the sparsity of the words, as there might be few common words

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
### 3. Suggest possible improvements to your pipeline

Instead of building a simple collection of unigrams (n=1), I built a collection of bigrams (n=2), where occurrences of pairs of consecutive words are counted.

with 2 ngrams
<421215x152044 sparse matrix of type '<class 'numpy.int64'>'
	with 7022630 stored elements in Compressed Sparse Row format>

A possible improvement would be to ...

Another potential improvement could be to ...
