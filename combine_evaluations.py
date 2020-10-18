import os
import json
import pandas as pd

model_folders = ["baseline", "categorical_and_pca", "categorical_only",
                 "combat_overfit", "reduce_float_dims_pca", "restrict_features",
                 "adding_features_v1", "adding_features_v2", "adding_features_v3"]

def combine_evaluations():
    evaluations = {}
    all_prfs = {}
    for model_folder in model_folders:
        with open(f"./{model_folder}/evaluation/evaluation.json", "r") as f:
            evaluation = json.load(f)
            evaluation.pop('test_target_proportions')
            prfs = evaluation.pop('prfs')
            all_prfs[model_folder] = prfs
            evaluations[model_folder] = evaluation
    prfs = pd.DataFrame(all_prfs).T
    evaluations = pd.DataFrame(evaluations).T.sort_values(by="accuracy", ascending=False)
    return {
        "prfs": prfs,
        "evaluations": evaluations
    }

if __name__ == "__main__":
    all_evaluations = combine_evaluations()
    all_evaluations["evaluations"].to_csv("evaluations.csv")
    all_evaluations["prfs"].to_csv("prfs.csv")
