# Patient-Survival-Prediction-using-Deep-Learning
Deep learning based modeling approach to train large dataset and predict patient survival 

**🧾Description:** Getting a rapid understanding of the context of a patient’s overall health has been particularly important during the COVID-19 pandemic as healthcare workers around the world struggle with hospitals overloaded by patients in critical condition. Intensive Care Units (ICUs) often lack verified medical histories for incoming patients. A patient in distress or a patient who is brought in confused or unresponsive may not be able to provide information about chronic conditions such as heart disease, injuries, or diabetes. Medical records may take days to transfer, especially for a patient from another medical provider or system. Knowledge about chronic conditions can inform clinical decisions about patient care and ultimately improve patient's survival outcomes.

**Source of the dataset:** [Click Here](https://journals.lww.com/ccmjournal/Citation/2019/01001/33__THE_GLOBAL_OPEN_SOURCE_SEVERITY_OF_ILLNESS.36.aspx)

**🧭 Problem Statement:** The target feature is hospital_death which is a binary variable. The task is to classify this variable based on the other 185 features step-by-step by going through each day's task. The scoring metric is Accuracy/Area under ROC curve.

**Key challange:** 

1. Large dataset with over 90K rows and 186 columns.
2. Lack of significant  relations with target feature
