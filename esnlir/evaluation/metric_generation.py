"""Module to evaluate metric statics"""
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from esnlir.evaluation.config import AVERAGING, METRICS

class MetricGenerator:
    def __init__(
        self,
        true_labels: list,
        predicted_labels: list,
        datasets: list,
        genres: list,
        domains: list,
        data_split: str,
        classes: list[str]
    ):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.data_split = data_split
        self.classes = classes

        self.df_groups = pd.DataFrame({
            "dataset": datasets,
            "genre": genres,
            "domain": domains,
            "true_label": true_labels,
            "predicted_label": predicted_labels
        })

    def evaluate_labels(self, predicted_labels, true_labels, data_split):
        metric_values = {}
        for metric_name, metric in METRICS.items():
            try:
                metric_values[metric_name] = [metric(
                    true_labels,
                    predicted_labels,
                    average=AVERAGING
                )]
            except:
                metric_values[metric_name] = [metric(
                    true_labels,
                    predicted_labels
                )]
        df_metrics = pd.DataFrame(metric_values)
        df_metrics["data_split"] = data_split
        return df_metrics
    
    def evaluate_grouped_labels(self, df_labels, group_column, data_split):
        group_results = []
        for group, df_group in df_labels.groupby(group_column):
            true_labels = df_group["true_label"].tolist()
            predicted_labels = df_group["predicted_label"].tolist()
            df_metrics = self.evaluate_labels(
                predicted_labels,
                true_labels,
                data_split
            )
            df_metrics[group_column] = group
            group_results.append(df_metrics)
        return pd.concat(group_results).sort_values("accuracy", ascending=False)

    def evaluate_class_accuracy(self, predicted_labels, true_labels, data_split, classes):
        conf_mat = confusion_matrix(true_labels, predicted_labels)
        class_accuracies = conf_mat.diagonal()/conf_mat.sum(axis=1)
        df_accuracies = pd.DataFrame({
            classes[ix]: [score]
            for ix, score in enumerate(class_accuracies)  
        })
        df_accuracies["data_split"] = data_split
        return df_accuracies

    def evaluate_classification_report(self, predicted_labels, true_labels, data_split, classes):
        # labels= keeps every class in the report even when the model never predicts it
        report = classification_report(
            true_labels,
            predicted_labels,
            labels=list(range(len(classes))),
            target_names=classes,
            output_dict=True,
            zero_division=0
        )
        df_report = pd.DataFrame(
            {name: values for name, values in report.items() if isinstance(values, dict)}
        ).T
        if "accuracy" in report:
            df_report.loc["accuracy", "f1-score"] = report["accuracy"]
            df_report.loc["accuracy", "support"] = len(true_labels)
        df_report["data_split"] = data_split
        return df_report

    def evaluate_grouped_class_accuracy(self, df_labels, group_column, data_split, classes):
        group_results = []
        for group, df_group in df_labels.groupby(group_column):
            true_labels = df_group["true_label"].tolist()
            predicted_labels = df_group["predicted_label"].tolist()
            df_metrics = self.evaluate_class_accuracy(
                predicted_labels,
                true_labels,
                data_split,
                classes
            )
            df_metrics[group_column] = group
            group_results.append(df_metrics)
        df_metrics = pd.concat(group_results)
        df_metrics["mean_accuracy"] = df_metrics[classes].mean(axis=1, skipna=True)
        return df_metrics.sort_values("mean_accuracy", ascending=False)
    
    def run(self):
        conf_mat = confusion_matrix(self.true_labels, self.predicted_labels)
        df_conf = pd.DataFrame(
            data=conf_mat,
            index=self.classes,
            columns=self.classes
        )
        metric_results = {
            f"{self.data_split}/total/confusion_matrix": df_conf,
            f"{self.data_split}/total/labels": pd.DataFrame({
                "true_label": [self.classes[label] for label in self.true_labels],
                "predicted_label": [self.classes[label] for label in self.predicted_labels]
            }),
            f"{self.data_split}/total/general_stats": self.evaluate_labels(
                self.predicted_labels,
                self.true_labels,
                self.data_split
            ),
            f"{self.data_split}/total/class_accuracy": self.evaluate_class_accuracy(
                self.predicted_labels,
                self.true_labels,
                self.data_split,
                self.classes
            ),
            f"{self.data_split}/total/classification_report": self.evaluate_classification_report(
                self.predicted_labels,
                self.true_labels,
                self.data_split,
                self.classes
            )
        }
        for aggrupation in ["dataset", "genre", "domain"]:
            aggrupation_stats_key = f"{self.data_split}/{aggrupation}/general_stats"
            df_aggrupation_stats = self.evaluate_grouped_labels(
                self.df_groups,
                aggrupation,
                self.data_split
            )
            metric_results[aggrupation_stats_key] = df_aggrupation_stats

            aggrupation_class_key = f"{self.data_split}/{aggrupation}/class_accuracy"
            df_aggrupation_class_accuracy = self.evaluate_grouped_class_accuracy(
                self.df_groups,
                aggrupation,
                self.data_split,
                self.classes
            )
            metric_results[aggrupation_class_key] = df_aggrupation_class_accuracy

        return metric_results