import os

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from utils import cluster, model_helper


class Prediction:

    def __init__(self, dataset, model, top_k=10, core="AE", mode="instance", batch_size=256, set_type="val"):
        self.dataset = dataset
        self.batch_size, self.top_k, self.core, self.mode = batch_size, top_k, core, mode
        self.set_type, self.model = set_type, model
        self.source_labels = []

    def run_model(self, loader, input_source=True):
        output_df = pd.DataFrame(columns=["code", "label"])
        for data, labels in loader:
            code, _ = model_helper.run_batch(self.model, data, self.core, False, input_source=input_source)
            code = [o.reshape(1, -1).squeeze(0) for o in code.cpu().numpy()]
            output_df = output_df.append(pd.DataFrame({"code": code, "label": labels}), ignore_index=True)
        return output_df

    @staticmethod
    def run_cluster(source_outputs, with_cluster):
        source_centers, source_labels_mapping = [], []
        if with_cluster:
            source_outputs = cluster.get_cluster_output_df(source_outputs, add_center=False)
            for (label, center), group_df in source_outputs.groupby(["label", "center"]):
                group_df = group_df.fillna(0)
                output = [o for o in group_df["code"]]
                source_centers.append(np.mean(output, axis=0))
                source_labels_mapping.append(label+str(center))
        else:
            for label, group_df in source_outputs.groupby(["label"]):
                output = [o for o in group_df["code"]]
                source_centers.append(np.mean(output, axis=0))
                source_labels_mapping.append(label)
        return source_centers, source_labels_mapping

    def get_classifier(self, with_cluster=False, source_mode="category", source_outputs=pd.DataFrame()):
        if source_outputs.empty:
            source_outputs = self.run_model(self.dataset.source_full_loader, input_source=True)
        if source_mode == "category":
            source_centers, self.source_labels = self.run_cluster(source_outputs, with_cluster)
        else:
            source_centers, self.source_labels = source_outputs["code"].tolist(), source_outputs["label"].tolist()
        source_centers = np.nan_to_num(source_centers)
        classifier = KNeighborsClassifier(n_neighbors=self.top_k)
        classifier.fit(source_centers, self.source_labels)
        return classifier

    def set_model(self, model):
        self.model = model

    def get_target(self):
        target_outputs = self.run_model(self.dataset.target_loader, input_source=False)
        target_centers, target_labels = [], []
        for label, group_df in target_outputs.groupby(["label"]):
            group_df = group_df.fillna(0)
            output = [o for o in group_df["code"]]
            if self.mode == "instance":
                target_centers.extend(output)
                target_labels.extend([label for _ in group_df["label"]])
            else:
                target_centers.append(np.mean(output, axis=0).tolist())
                target_labels.append(label)
        return target_outputs, target_centers, target_labels

    def _get_source_mapping(self, i):
        label = self.source_labels[i]
        if len(label) < 4:
            label = label
        else:
            label = label.split(os.sep)[-3]
        return label

    def get_result(self, target_centers, target_labels, classifier):
        count, index_sum = 0, 0
        correct_char = {}
        top_n_chars = classifier.kneighbors(target_centers, return_distance=False)
        for top_n_char, target_label in zip(top_n_chars, target_labels):
            predicted_chars = [self._get_source_mapping(i) for i in top_n_char]
            if target_label in set(predicted_chars):
                count += 1
                correct_index = predicted_chars.index(target_label)  # the rank of target input
                index_sum += correct_index
                if correct_index not in correct_char:
                    correct_char[correct_index] = []
                correct_char[correct_index].append(target_label)
        return correct_char, count, index_sum

    def predict(self, classifier=None):
        if classifier is None:
            classifier = self.get_classifier(False)
        target_outputs, target_centers, target_labels = self.get_target()
        correct_char, count, index_sum = self.get_result(target_centers, target_labels, classifier)
        if self.mode == "instance":
            accuracy = count / len(target_outputs)
        else:
            accuracy = count / len(target_outputs.groupby(["label"]))
        # sort prediction result
        correct_char = {k: v for k, v in sorted(correct_char.items(), key=lambda j: j[0])}
        chars = list()
        for c in correct_char.values():
            chars.extend(c)
        keys = ["accuracy", "index_sum", "correct", "chars"]
        values = [accuracy, index_sum, correct_char, sorted(set(chars))]
        return {"%s_%s" % (self.set_type, k): v for k, v in zip(keys, values)}
