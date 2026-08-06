"""Active Learning Trainer that orchestrates acquisition cycles.

Reuses the project's HF Trainer and dataset. Warm-start by default.
Logs selected/removed indices and metrics per iteration under output directory.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoModelForSequenceClassification

from .pool_manager import PoolManager
from .base_strategy import ActiveLearningStrategy
from esnlir.evaluation.metric_generation import MetricGenerator
from esnlir.utils.utils import stratified_indices


@dataclass
class ActiveLearningConfig:
    L: int = 500  # iterations
    K: int = 8    # acquisitions per iter
    remove_k: int | None = None
    scoring_batch_size: int = 64
    warm_start: bool = True
    # Global convergence controls across AL iterations (optional)
    # Stop when no acquisitions are made in an iteration (empty selection)
    stop_if_no_acquisition: bool = True
    # Early stopping across iterations based on validation metric
    iter_patience: int | None = None  # number of iters with no improvement to stop (None = disabled)
    iter_min_delta: float = 0.0       # required improvement margin
    iter_metric: str = "f1_score"     # one of the keys returned by compute_metrics_fn (e.g., 'f1_score' or 'accuracy')
    # Per-cycle per-class report over a fixed, class-balanced subset of validation.
    # Kept small so it stays cheap; 0 disables, None uses all of validation.
    report_n: int | None = 2000
    # Test splits to evaluate on every cycle (None = all). The full set is always
    # evaluated once at the end, so restricting this only trims per-cycle cost.
    eval_splits: list[str] | None = None


class ActiveLearningTrainer:
    def __init__(
        self,
        model_name: str,
        base_output_dir: str,
        strategy: ActiveLearningStrategy,
        pool_manager: PoolManager,
        full_train_dataset,
        val_dataset,
        test_datasets: dict,
        n_classes: int,
        class_weights,
        training_args: TrainingArguments,
        early_stopping_patience: int,
        device: str = "cpu",
        al_config: Optional[ActiveLearningConfig] = None,
        compute_metrics_fn=None
    ):
        self.model_name = model_name
        self.base_output_dir = base_output_dir
        self.strategy = strategy
        self.pool = pool_manager
        self.full_train_dataset = full_train_dataset
        self.val_dataset = val_dataset
        self.test_datasets = test_datasets
        self.n_classes = n_classes
        self.class_weights = class_weights
        self.early_stopping_patience = early_stopping_patience
        self.compute_metrics_fn = compute_metrics_fn
        self.device = torch.device(device)
        self.alc = al_config or ActiveLearningConfig()
        self.training_args = training_args
        # Sampled once and reused every cycle, so per-class scores stay comparable across iterations
        self.val_report_indices = self._build_val_report_subset(
            self.alc.report_n,
            training_args.seed
        )

    def _build_val_report_subset(self, report_n, seed):
        """Fixed, class-balanced subset of validation used for the per-cycle report."""
        if report_n is None:
            return None
        if report_n <= 0:
            return []
        return stratified_indices(
            self.val_dataset.targets,
            report_n,
            self.n_classes,
            seed
        )

    def _report_validation(self, trainer, out_root, it):
        """Print and save a per-class report on the validation subset."""
        if self.val_report_indices == []:
            return
        dataset = (
            self.val_dataset
            if self.val_report_indices is None
            else Subset(self.val_dataset, self.val_report_indices)
        )
        model_output = trainer.predict(dataset)
        y_true = model_output.label_ids.argmax(axis=1)
        y_pred = model_output.predictions.argmax(axis=1)
        classes = self.val_dataset.classes
        label_ids = list(range(len(classes)))

        print(f"\n=== Validation report | iter {it} | n={len(y_true)} ===")
        print(classification_report(
            y_true, y_pred,
            labels=label_ids, target_names=classes, zero_division=0
        ))
        # Acquired-class balance: shows whether the strategy is skewing the labeled set
        acquired = np.asarray(self.full_train_dataset.targets)[self.pool.labeled_indices].argmax(axis=1)
        counts = dict(zip(*np.unique(acquired, return_counts=True)))
        print("labeled pool by class:", {classes[c]: int(n) for c, n in counts.items()}, flush=True)

        report = classification_report(
            y_true, y_pred,
            labels=label_ids, target_names=classes,
            output_dict=True, zero_division=0
        )
        df_report = pd.DataFrame(
            {name: values for name, values in report.items() if isinstance(values, dict)}
        ).T
        df_report["iteration"] = it
        df_report["n_labeled"] = len(self.pool.labeled_indices)
        df_report.to_csv(os.path.join(out_root, f"val_report_iter_{it}.csv"))

    def _make_unlabeled_loader(self, batch_size: int):
        subset = self.pool.get_unlabeled_subset(self.full_train_dataset)
        # Scoring tokenises on the fly too, so reuse the configured worker count
        return DataLoader(
            subset,
            batch_size=batch_size,
            num_workers=self.training_args.dataloader_num_workers
        )

    def _make_labeled_subset(self):
        return self.pool.get_labeled_subset(self.full_train_dataset)

    def _build_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.n_classes
        )
        return model.to(self.device)

    def _build_trainer(self, model, train_dataset):
        # Import here to reuse existing WeightedTrainer if available
        from esnlir.training.train import WeightedTrainer
        callbacks = []
        if self.early_stopping_patience is not None:
            callbacks = [EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience)]
        trainer = WeightedTrainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics_fn,
            #callbacks=callbacks,
            class_weights=self.class_weights
        )
        return trainer

    def _evaluate_and_log(self, trainer: Trainer, out_dir: str, splits: list[str] | None = None):
        os.makedirs(out_dir, exist_ok=True)
        if splits is not None:
            unknown = set(splits) - set(self.test_datasets)
            if unknown:
                print(
                    f"WARNING: eval_splits {sorted(unknown)} match no test split; "
                    f"available: {sorted(self.test_datasets)}",
                    flush=True
                )
        # Evaluate on all test splits using existing util
        for test_split, test_loader in self.test_datasets.items():
            if splits is not None and test_split not in splits:
                continue
            model_output = trainer.predict(test_loader)
            metric_generator = MetricGenerator(
                model_output.label_ids.argmax(axis=1),
                model_output.predictions.argmax(axis=1),
                test_loader.datasets,
                test_loader.genres,
                test_loader.domains,
                test_split,
                test_loader.classes
            )
            metrics = metric_generator.run()
            for path, df_result in metrics.items():
                saving_path = os.path.join(out_dir, f"{path}.csv")
                saving_folder = "/".join(saving_path.split("/")[:-1])
                if not os.path.exists(saving_folder):
                    os.makedirs(saving_folder)
                df_result.to_csv(saving_path)

    def _evaluate_best_model(self, out_root: str, best_iter, last_it):
        """Evaluate the saved best checkpoint over all test splits.

        Skipped when the best iteration is also the last one, since final/ then already
        describes the same weights.
        """
        if best_iter is None:
            return
        if best_iter == last_it:
            print(
                f"Best iteration ({best_iter}) is the final one; "
                f"final/ already describes best_model",
                flush=True
            )
            return
        best_dir = os.path.join(out_root, "best_model")
        if not os.path.exists(best_dir):
            return
        print(f"Evaluating best_model (iter {best_iter}) over all test splits", flush=True)
        best_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(self.device)
        self._evaluate_and_log(
            self._build_trainer(best_model, None),
            os.path.join(out_root, "best")
        )

    def run(self, strategy_name: str):
        out_root = os.path.join(self.base_output_dir, f"{self.model_name.split('/')[-1]}_active_{strategy_name}")
        os.makedirs(out_root, exist_ok=True)

        # Initialize model once; warm-start means we keep weights across cycles
        model = self._build_model()
        last_trainer = None
        # If starting with some labeled indices, do an initial training pass
        if self.pool.labeled_indices:
            labeled_ds = self._make_labeled_subset()
            trainer = self._build_trainer(model, labeled_ds)
            trainer.train()
            last_trainer = trainer
            self._evaluate_and_log(trainer, os.path.join(out_root, "init"), splits=self.alc.eval_splits)

        best_iter_metric = None
        best_iter = None
        last_it = None
        no_improve = 0

        for it in range(self.alc.L):
            # Scoring over unlabeled pool
            unl_loader = self._make_unlabeled_loader(self.alc.scoring_batch_size)
            selection = self.strategy.select(model.eval(), unl_loader, self.alc.K)

            removed_indices = []
            selected_indices = []
            if isinstance(selection, tuple):
                selected_indices, removed_indices = selection
            else:
                selected_indices = selection

            # Update pools
            if removed_indices:
                self.pool.remove(removed_indices)
            if selected_indices:
                self.pool.acquire(selected_indices)

            # Optional: stop when nothing was acquired (e.g., pool empty or thresholded selection)
            if self.alc.stop_if_no_acquisition and len(selected_indices) == 0:
                # Nothing to add; consider this converged
                break

            # Log indices
            pd.DataFrame({"index": selected_indices}).to_csv(
                os.path.join(out_root, f"selected_indices_iter_{it}.csv"), index=False
            )
            if removed_indices:
                pd.DataFrame({"index": removed_indices}).to_csv(
                    os.path.join(out_root, f"removed_indices_iter_{it}.csv"), index=False
                )

            # Retrain using labeled subset (warm-start by default)
            labeled_ds = self._make_labeled_subset()
            # If not warm-start, reset model to pretrained each iteration
            if not self.alc.warm_start:
                model = self._build_model()
            trainer = self._build_trainer(model, labeled_ds)
            trainer.train()
            last_trainer = trainer
            last_it = it

            self._report_validation(trainer, out_root, it)

            # Evaluate and save basic summary per iter
            iter_dir = os.path.join(out_root, f"iter_{it}")
            self._evaluate_and_log(trainer, iter_dir, splits=self.alc.eval_splits)
            # Save metrics.json (aggregate from eval dataset evaluate)
            eval_metrics = trainer.evaluate(self.val_dataset)
            # Flatten and keep only eval_* metrics
            iter_metrics = {k: float(v) for k, v in eval_metrics.items() if k.startswith("eval_")}
            with open(os.path.join(out_root, f"metrics_iter_{it}.json"), "w", encoding="utf-8") as f:
                json.dump(iter_metrics, f, indent=2)

            # Track the best iteration independently of early stopping, so the snapshot is
            # taken even when iter_patience is disabled.
            # Trainer.evaluate returns keys like 'eval_f1_score'; map from requested metric
            key = f"eval_{self.alc.iter_metric}"
            current = iter_metrics.get(key)
            if current is not None:
                improved = (
                    (best_iter_metric is None)
                    or (current > best_iter_metric + self.alc.iter_min_delta)
                )
                if improved:
                    best_iter_metric = current
                    best_iter = it
                    no_improve = 0
                    best_dir = os.path.join(out_root, "best_model")
                    trainer.save_model(best_dir)
                    with open(os.path.join(best_dir, "best_iteration.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "iteration": it,
                            "metric": key,
                            "value": current,
                            "n_labeled": len(self.pool.labeled_indices)
                        }, f, indent=2)
                    print(f"New best {key}={current:.4f} at iter {it} -> snapshot saved", flush=True)
                else:
                    no_improve += 1
                # Global early stopping across iterations (optional)
                if self.alc.iter_patience is not None and no_improve >= int(self.alc.iter_patience):
                    print(f"Stopping: {key} did not improve for {no_improve} iterations", flush=True)
                    break

        # Save final model at the end of AL run (either by exhaustion, early stop, or after L iterations)
        if last_trainer is not None:
            # Per-cycle evaluation may be restricted via eval_splits; evaluate everything once here
            print("Final evaluation over all test splits", flush=True)
            self._evaluate_and_log(last_trainer, os.path.join(out_root, "final"))
            final_dir = os.path.join(out_root, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            last_trainer.save_model(final_dir)
            if best_iter is not None:
                print(
                    f"Best iteration: {best_iter} "
                    f"(eval_{self.alc.iter_metric}={best_iter_metric:.4f}) -> {out_root}/best_model",
                    flush=True
                )
            self._evaluate_best_model(out_root, best_iter, last_it)
