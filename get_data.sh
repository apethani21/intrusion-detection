TASK=http://kdd.ics.uci.edu/databases/kddcup99/task.html
KDDCUP_NAMES=http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
FULL_DATA=http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz
LABELLED_TEST_DATA=http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz
ATTACK_TYPES=http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types
TYPO_CORRECTION=http://kdd.ics.uci.edu/databases/kddcup99/typo-correction.txt


curl -o ./raw-data/task.html $TASK
curl -o ./raw-data/feature_names.txt $KDDCUP_NAMES
curl -o ./raw-data/full_data.gz $FULL_DATA
curl -o ./raw-data/test_labelled.gz $LABELLED_TEST_DATA
curl -o ./raw-data/attack_types.txt $ATTACK_TYPES
curl -o ./raw-data/typo-correction.txt $TYPO_CORRECTION
