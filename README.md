# browser_family_extraction
Use machine learning models to predict browser family and version number

---

## Overview

Report is written as markdown file under writeup.md

There is one notebook (demo.ipynb) created for visualization and showing the analysis process.

To look into the notebook, you would need to cd to the repo directory and run
```
cd browser_family_extraction
jupyter notebook demo.ipynb
```

Code is under prediction_browser_family.py

To run the script, you would need to cd to the repo directory and run

```
python predict_browser_family.py --training data_coding_exercise.txt --test test_data_coding_exercise.txt --prediction-results test_results.txt
```

Stored test results are at `results/test_results.txt`

To test the script on new data set, you would need to copy the new txt file to `datasets` folder.

---