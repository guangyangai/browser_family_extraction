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
Given the size and nature of the data, I decided to use batch, model-based learning.
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

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
