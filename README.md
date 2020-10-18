# Intrusion detection - KDD Cup 1999

Intrusion detection using the [1999 KDD Cup dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html). kddcup.data.gz is used for training/validation and corrected.gz is used as the test set, which includes attack types which are not in the training data to more closely align with the challenges of intrusion detection in practice.

- `get_data.sh` downloads the data (data not included in repository).
- `explore_data.ipynb` contains some exploratory work, most importantly finding the presence of duplicated rows, a column whose value was constant and not valuable in this task, and class imbalance (as expected with intrusion detection).
- `base.py` contains the `ModelHelper` class, which handles routines such as preparing training and test data, training, hyperparameter-tuning, model loading & saving, and model evaluation.
- `kdd_env.yml` contains the dependencies for this project.
