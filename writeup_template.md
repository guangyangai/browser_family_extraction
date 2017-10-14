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
I decided to use text feature extraction modules in sklearn and applied on agent description to create

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
