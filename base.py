import os
import json
import copy
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             precision_recall_fscore_support,
                             classification_report,
                             plot_confusion_matrix,
                             roc_auc_score,
                             roc_curve)


class ModelHelper:
    def __init__(self,
                 model_folder,
                 model_name,
                 path_to_full_data="./raw-data/full_data.gz",
                 path_to_test_data="./raw-data/test_labelled.gz",
                 seed=1):
        self.model_name = model_name
        self.model_folder = model_folder
        self.pipeline_metadata_folder = f"{self.model_folder}/metadata"
        self.evaluation_folder = f"{self.model_folder}/evaluation"
        self.seed = seed
        self.path_to_full_data = path_to_full_data
        self.path_to_test_data = path_to_test_data
        self.path_to_feature_names = "./raw-data/feature_names.txt"
        self.model = None
        self.exhaustive_grid_search_result = None
        self.random_grid_search_results = None
        self.evaluation = None
        self._initialise_folders()
        self._default_param_grid = {
             "n_estimators": [50, 75, 100],
             "max_depth": [2, 4, 6, 8],
             "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
             "gamma": [0, 2.5, 5],
             "subsample": [0.5, 0.75, 1]
        }
        self.best_params = None

    def __repr__(self):
        return (f"Model Folder = {self.model_folder}\n"
                f"Metadata Folder = {self.pipeline_metadata_folder}\n"
                f"Path to data = {self.path_to_full_data}\n"
                f"Path to test data = {self.path_to_test_data}\n"
                f"Path to evaluation folder = {self.evaluation_folder}\n"
                f"Seed = {self.seed}")

    def _initialise_folders(self):
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.pipeline_metadata_folder, exist_ok=True)
        os.makedirs(self.evaluation_folder, exist_ok=True)

    def _create_service_encoding(self, df):
        label_encoder = LabelEncoder()
        label_encoder.fit(df["service"])
        with open(f"{self.pipeline_metadata_folder}/service_encoding.pkl", 'wb') as f:
            pickle.dump(label_encoder, f)
        return

    @staticmethod
    def _apply_encoder(encoder, series):
        try:
            return encoder.transform(series)
        except ValueError:
            encoder_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            return series.apply(lambda x: encoder_dict.get(x, -1))

    @staticmethod
    def _target_binary_encoding(target):
        return not target == "normal."

    def transform_categorical(self, df):
        df["protocol_type"] = df["protocol_type"].apply(lambda x: x.lower().strip())
        df["flag"] = df["flag"].apply(lambda x: x.strip())
        df = pd.get_dummies(df, columns=['protocol_type', 'flag'])
        service_encoding_file = f"{self.pipeline_metadata_folder}/service_encoding.pkl"
        if not os.path.isfile(service_encoding_file):
            print("service_encoding.pkl missing. creating now.")
            self._create_service_encoding(df)
        with open(service_encoding_file, "rb") as f:
            service_encoder = pickle.load(f)
        df['service'] = self._apply_encoder(service_encoder, df['service'])
        return df

    def prepare_data(self, path_to_data):
        df = pd.read_csv(path_to_data, header=None)
        df.drop_duplicates(inplace=True, ignore_index=True)

        with open(self.path_to_feature_names, "r") as f:
            feature_names_contents = f.read().split("\n")
        feature_names_contents = feature_names_contents[1:-1]
        columns_names = [name.split(":")[0].strip().lower()
                         for name in feature_names_contents]
        columns_names.append("target")
        df.columns = columns_names
        df = df.loc[:, df.apply(pd.Series.nunique) != 1]
        df = self.transform_categorical(df)
        df["target"] = df["target"].apply(self._target_binary_encoding)
        return df

    def exhaustive_grid_search(self,
                               X_train,
                               y_train,
                               param_grid=None,
                               fit_params=None,
                               cv=None,
                               model=None):
        fit_params = fit_params or {}
        if model is None and self.model is None:
            raise ValueError("Both model argument and model property are None")
        if hasattr(cv, 'random_state'):
            cv.random_state = self.seed
        model = model or self.model
        if isinstance(model, xgboost.sklearn.XGBClassifier):
            param_grid = param_grid or self._default_param_grid

        grid_searcher = GridSearchCV(
            estimator=model, param_grid=param_grid,
            scoring='accuracy', verbose=25,
            refit=False, n_jobs=-1, cv=cv
        )
        grid_searcher.fit(X_train, y_train, verbose=True, **fit_params)
        self.exhaustive_grid_search_result = copy.deepcopy(grid_searcher)
        self.best_params = copy.deepcopy(grid_searcher.best_params_)
        with open(f"./{self.pipeline_metadata_folder}/grid_searcher_result.pkl", "wb") as f:
            pickle.dump(grid_searcher, f)
        return grid_searcher.best_params_

    def random_grid_search(self,
                           X_train,
                           y_train,
                           param_distribution=None,
                           fit_params=None,
                           cv=None,
                           model=None,
                           n_iter=None):
        fit_params = fit_params or {}
        if model is None and self.model is None:
            raise ValueError("Both model argument and model property are None")
        if hasattr(cv, 'random_state'):
            cv.random_state = self.seed
        model = model or self.model
        if isinstance(model, xgboost.sklearn.XGBClassifier):
            param_distribution = param_distribution or self._default_param_grid

        random_grid_searcher = RandomizedSearchCV(
            estimator=model, param_distributions=param_distribution,
            scoring='accuracy', verbose=25, n_iter=n_iter,
            refit=False, n_jobs=-1, cv=cv, random_state=self.seed
        )
        random_grid_searcher.fit(X_train, y_train, verbose=True, **fit_params)
        self.random_grid_search_result = copy.deepcopy(random_grid_searcher)
        self.best_params = copy.deepcopy(random_grid_searcher.best_params_)
        with open(f"./{self.pipeline_metadata_folder}/random_grid_searcher_result.pkl", "wb") as f:
            pickle.dump(random_grid_searcher, f)
        return random_grid_searcher.best_params_

    def plot_importance(self):
        assert isinstance(self.model, xgboost.sklearn.XGBClassifier)
        with plt.style.context('seaborn'):
            fig, axes = plt.subplots(1, 3, figsize=(18, 15))
            fig.tight_layout(pad=8)
            axes[0] = xgboost.plot_importance(self.model, ax=axes[0],
                                              importance_type="weight",
                                              xlabel="weight")
            axes[1] = xgboost.plot_importance(self.model, ax=axes[1],
                                              importance_type="gain",
                                              xlabel="gain")
            axes[2] = xgboost.plot_importance(self.model, ax=axes[2],
                                              importance_type="cover",
                                              xlabel="cover")
            for ax, title in zip(axes, ["weight", "gain", "cover"]):
                ax.set_title(f"Features ranked by {title}")
                ax.set_ylabel(None)
                ax.tick_params(axis="x", labelrotation=45, labelsize=10)
                ax.tick_params(axis="y", labelrotation=45, labelsize=10)
                for txt_obj in ax.texts:
                    curr_text = txt_obj.get_text()
                    new_text = round(float(curr_text), 1)
                    txt_obj.set_text(new_text)
            fig.savefig(f"{self.evaluation_folder}/importance.png", dpi=300)
        return

    def plot_tree(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 30))
        xgboost.plot_tree(self.model, ax=ax)
        fig.savefig(f"{self.evaluation_folder}/tree.png", dpi=500)

    def plot_history(self):
        assert isinstance(self.model, xgboost.sklearn.XGBClassifier)
        with plt.style.context('seaborn'):
            history = self.model.evals_result()
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.plot(range(len(history['validation_0']['error'])),
                    history['validation_0']['error'],
                    label="Training error", linewidth=1.5)
            ax.plot(range(len(history['validation_1']['error'])),
                    history['validation_1']['error'],
                    label="Testing error", linewidth=1.5)
            ax.set_title('Error')
            ax.grid(True)
            ax.legend()
            fig.savefig(f"{self.evaluation_folder}/train_test_error.png", dpi=300)
        return

    def evaluate_model(self, X_test, y_test, save_evaluation=True):
        if self.model is None:
            raise ValueError("Cannot evaluate model of type None")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred, adjusted=True)
        test_target_proportions = y_test.value_counts(normalize=True).to_dict()
        prfs = precision_recall_fscore_support(y_test, y_pred)
        prfs = [dict(zip(["normal", "attack"], list(map(float, arr)))) for arr in prfs]
        prfs = dict(zip(["precision", "recall", "f1_score", "support"], prfs))
        print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))

        with plt.style.context('seaborn'):
            if isinstance(self.model, xgboost.sklearn.XGBClassifier):
                self.plot_importance()
                self.plot_history()
                self.plot_tree()

            plt.figure(figsize=(8, 8))
            fpr, tpr, _ = roc_curve(y_test, self.model.predict_proba(X_test)[:, 1])
            auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
            plt.plot(fpr, tpr, color='darkorange', linewidth=1.5)
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC (Area = {round(auc, 3)})')
            plt.grid(True)
            plt.savefig(f"{self.evaluation_folder}/roc.png")

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            plot_confusion_matrix(self.model, X_test, y_test,
                                  display_labels=["normal", "attack"],
                                  cmap=plt.cm.copper, normalize='true', ax=ax)
            fig.savefig(f"{self.evaluation_folder}/confusion_matrix.png", dpi=300)

        evaluation = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "prfs": prfs,
            "test_target_proportions": test_target_proportions,
            "model_params": self.model.get_params()
        }

        self.evaluation = evaluation
        if save_evaluation:
            with open(f"{self.evaluation_folder}/evaluation.json", "w") as f:
                json.dump(evaluation, f, indent=4)

        return evaluation

    def save_model(self):
        if self.model is None:
            raise ValueError("Cannot save model of type None")
        with open(f"{self.model_folder}/{self.model_name}.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, set_model_attr=True):
        with open(f"{self.model_folder}/{self.model_name}.pkl", "rb") as f:
            model = pickle.load(f)
        if set_model_attr:
            self.model = model
        return model
