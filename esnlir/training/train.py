"""Module to train BERT based models

Adds optional Active Learning flow via --active-learning flag.
"""
import os
import json
import argparse
import torch

from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer
)

from torch.utils.data import DataLoader

# Active Learning imports
from esnlir.active_learning import (
    ActiveLearningTrainer,
    ActiveLearningConfig,
    PoolManager,
    RandomStrategy,
    NegativeEnergyStrategy,
    RemStrategy,
)

from esnlir.dataset_utils.dataset import BERTDataset 
from esnlir.utils.utils import seed_everything, stratified_indices
from esnlir.evaluation.config import AVERAGING, METRICS

from esnlir.evaluation.metric_generation import MetricGenerator

def eval_model(
    pred
):
    predicted = torch.tensor(pred.predictions)
    gold_standard = torch.tensor(pred.label_ids)
    metric_values = {}
    with torch.no_grad():
        y_true = torch.argmax(gold_standard, axis=1).to("cpu")
        y_pred = torch.argmax(predicted, axis=1).to("cpu")
        metric_values = {}
        for metric_name, metric in METRICS.items():
            try:
                metric_values[metric_name] = metric(
                    y_true,
                    y_pred,
                    average=AVERAGING
                )
            except:
                metric_values[metric_name] = metric(
                    y_true,
                    y_pred
                )
    return metric_values

def write_split_result(output_folder, metrics):
    for path, df_result in metrics.items():
        saving_path = os.path.join(output_folder, f"{path}.csv")
        saving_folder = "/".join(saving_path.split("/")[:-1])
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)
        df_result.to_csv(saving_path)

def evaluate_test_splits(output_folder, test_loaders):
    for test_split, test_loader in test_loaders.items():
        print(f"Evaluating split {test_split}")
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
        write_split_result(output_folder, metrics)


class WeightedTrainer(Trainer):

    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, num_items_in_batch=32, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights, device=logits.device)
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

if __name__ == '__main__':
    print("Starting model training and evaluation")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help="Configuration file path")
    # Active learning flags
    parser.add_argument("--active-learning", action="store_true", help="Enable Active Learning")
    parser.add_argument("--al_strategy", type=str, default=None, help="Active Learning strategy: NegE|Random|Rem")
    parser.add_argument("--al_L", type=int, default=None, help="AL iterations (L)")
    parser.add_argument("--al_K", type=int, default=None, help="AL acquisitions per iter (K)")
    parser.add_argument("--al_remove_k", type=int, default=None, help="AL removals per iter (REM)")
    parser.add_argument("--al_scoring_batch_size", type=int, default=None, help="Batch size for scoring")
    parser.add_argument("--al_warm_start", type=bool, default=None, help="Warm-start fine-tuning between cycles")
    parser.add_argument("--al_iter_patience", type=int, default=None, help="AL iteration-level early stopping patience")
    parser.add_argument("--al_iter_min_delta", type=float, default=None, help="AL iteration-level min delta for improvement")
    parser.add_argument("--al_iter_metric", type=str, default=None, help="AL iteration-level metric to monitor (f1_score|accuracy)")
    parser.add_argument("--al_top_p", type=float, default=None, help="AL strategy: select from top p proportion by score (0-1]")
    parser.add_argument("--al_min_energy", type=float, default=None, help="AL NegE: minimum energy threshold to acquire")
    parser.add_argument("--al_rem_select_mode", type=str, default=None, help="AL Rem: 'random' or 'uncertainty' selection")
    parser.add_argument("--al_rem_utility", type=str, default=None, help="AL Rem: 'energy' or 'entropy' utility")
    parser.add_argument("--al_seed_size", type=int, default=None, help="AL: size of the class-balanced initial labeled seed set")
    args = parser.parse_args()
    config_file = args.config_file
    with open(config_file, "r") as f:
        params = json.load(f)

    # Read core training params
    dataset_folder = params.get("dataset_folder")
    model_type = params.get("model_type")
    n_epochs = params.get("n_epochs")
    max_len = params.get("max_len")
    output_folder = params.get("output_folder")
    monitoring_metric = params.get("monitor")
    patience = params.get("patience")
    batch_size=params.get("batch_size")
    eval_batch_size = params.get("eval_batch_size", batch_size * 2)
    # Mixed precision and dataloader workers matter a lot on fast GPUs: BERTDataset
    # tokenises inside __getitem__, so with 0 workers the CPU starves the GPU.
    bf16 = params.get("bf16", False)
    fp16 = params.get("fp16", False)
    dataloader_num_workers = params.get("dataloader_num_workers", 0)
    only_premise = params.get("only_premise", False)
    warmup_steps = params.get("warmup_steps")
    random_seed = params.get("random_seed")
    learning_rate = params.get("learning_rate")
    device = params.get("device")
    max_samples = params.get("max_samples", None)

    # Optional AL config from JSON
    al_from_cfg = params.get("active_learning", False)
    al_strategy_cfg = params.get("al_strategy", None)
    al_L_cfg = params.get("al_L", None)
    al_K_cfg = params.get("al_K", None)
    al_remove_k_cfg = params.get("al_remove_k", None)
    al_scoring_bs_cfg = params.get("al_scoring_batch_size", None)
    al_warm_cfg = params.get("warm_start", True)
    al_iter_patience_cfg = params.get("al_iter_patience", None)
    al_iter_min_delta_cfg = params.get("al_iter_min_delta", 0.0)
    al_iter_metric_cfg = params.get("al_iter_metric", None)
    al_top_p_cfg = params.get("al_top_p", None)
    al_min_energy_cfg = params.get("al_min_energy", None)
    al_rem_select_mode_cfg = params.get("al_rem_select_mode", "random")
    al_rem_utility_cfg = params.get("al_rem_utility", "energy")
    al_report_n_cfg = params.get("al_report_n", 2000)
    al_eval_splits_cfg = params.get("al_eval_splits", None)
    al_seed_size_cfg = params.get("al_seed_size", 0)

    # Merge CLI > JSON defaults
    use_active_learning = args.active_learning or bool(al_from_cfg)
    al_strategy_name = args.al_strategy or al_strategy_cfg or "Random"
    al_L_val = args.al_L or al_L_cfg or 500
    al_K_val = args.al_K or al_K_cfg or 8
    al_remove_k_val = args.al_remove_k if args.al_remove_k is not None else (al_remove_k_cfg)
    al_scoring_bs = args.al_scoring_batch_size or al_scoring_bs_cfg or 64
    al_warm_start = args.al_warm_start if args.al_warm_start is not None else al_warm_cfg
    al_iter_patience = args.al_iter_patience if args.al_iter_patience is not None else al_iter_patience_cfg
    al_iter_min_delta = args.al_iter_min_delta if args.al_iter_min_delta is not None else al_iter_min_delta_cfg
    al_iter_metric = args.al_iter_metric or al_iter_metric_cfg or monitoring_metric or "f1_score"
    al_top_p = args.al_top_p if args.al_top_p is not None else al_top_p_cfg
    al_min_energy = args.al_min_energy if args.al_min_energy is not None else al_min_energy_cfg
    al_rem_select_mode = args.al_rem_select_mode or al_rem_select_mode_cfg
    al_rem_utility = args.al_rem_utility or al_rem_utility_cfg
    al_seed_size = args.al_seed_size if args.al_seed_size is not None else al_seed_size_cfg

    seed_everything(random_seed)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    train_loader = BERTDataset(
        os.path.join(dataset_folder, "train.json"),
        max_len,
        model_type,
        only_premise,
        max_samples
    )

    val_loader = BERTDataset(
        os.path.join(dataset_folder, "val.json"),
        max_len,
        model_type,
        only_premise,
        max_samples
    )

    test_loaders = {
        os.path.splitext(os.path.basename(test_file))[0]: BERTDataset(
            os.path.join(dataset_folder, test_file),
            max_len,
            model_type,
            only_premise,
            max_samples
        )
        for test_file in os.listdir(dataset_folder)
        if ("test" in test_file) and (".json" in test_file)
    }

    n_classes = train_loader.n_classes
    class_weights = train_loader.class_weights

    # AL reports on validation once per cycle, so in-training evaluation is only useful
    # for the plain path, where it also drives early stopping and best-model selection.
    eval_strategy = "no" if use_active_learning else "epoch"
    # Under AL nothing reads HF's per-epoch checkpoints: load_best_model_at_end is off and
    # there is no resume path. They are ~3.3 GB each (weights + AdamW state) and would be
    # rewritten every epoch of every cycle. best_model/ and final_model/ are saved explicitly.
    save_strategy = "no" if use_active_learning else "epoch"

    training_args = TrainingArguments(
        warmup_steps=warmup_steps,
        output_dir=output_folder,
        overwrite_output_dir=True,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        save_total_limit=1,
        seed=random_seed,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        metric_for_best_model=monitoring_metric,
        load_best_model_at_end=(not use_active_learning),
        report_to=["tensorboard"],
        learning_rate=learning_rate,
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_type,
        num_labels=n_classes
    )
    model.to(device)

    if not use_active_learning:
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_loader,
            eval_dataset=val_loader,
            compute_metrics=eval_model,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)],
            class_weights=class_weights
        )

        print("Training model")
        trainer.train()

        print("Evaluating in test")
        evaluate_test_splits(output_folder, test_loaders)
        print("Saving model")
        trainer.save_model(os.path.join(output_folder, "model"))
    else:
        # Build AL strategy
        strategy_name_norm = (al_strategy_name or "Random").lower()
        if strategy_name_norm in ["random", "pas", "rand", "rnd"]:
            strategy = RandomStrategy(seed=random_seed)
            resolved_strategy_name = "Random"
        elif strategy_name_norm in ["nege", "negenergy", "energy"]:
            strategy = NegativeEnergyStrategy(device=device, min_energy_threshold=al_min_energy, top_p=al_top_p)
            resolved_strategy_name = "NegE"
        elif strategy_name_norm in ["rem", "remove"]:
            strategy = RemStrategy(
                device=device,
                remove_k=al_remove_k_val or 0,
                utility=al_rem_utility,
                select_mode=al_rem_select_mode,
                seed=random_seed
            )
            resolved_strategy_name = "Rem"
        else:
            raise ValueError(f"Unknown AL strategy: {al_strategy_name}")

        # Start from a class-balanced seed set so the first acquisition is scored by a
        # trained head rather than a randomly initialised one. Seeded from random_seed,
        # so every strategy starts from the identical set.
        initial_labeled = []
        if al_seed_size:
            initial_labeled = stratified_indices(
                train_loader.targets,
                al_seed_size,
                n_classes,
                random_seed
            )
            print(f"AL seed set: {len(initial_labeled)} examples ({len(initial_labeled) // n_classes} per class)")
        pool = PoolManager(total_size=len(train_loader), initial_labeled_indices=initial_labeled)

        # Active Learning TrainingArguments: reuse but different output folder
        al_output_base = output_folder
        al_config = ActiveLearningConfig(
            L=int(al_L_val),
            K=int(al_K_val),
            remove_k=(int(al_remove_k_val) if al_remove_k_val is not None else None),
            scoring_batch_size=int(al_scoring_bs),
            warm_start=bool(al_warm_start),
            stop_if_no_acquisition=True,
            iter_patience=(int(al_iter_patience) if al_iter_patience is not None else None),
            iter_min_delta=float(al_iter_min_delta) if al_iter_min_delta is not None else 0.0,
            iter_metric=str(al_iter_metric),
            report_n=(int(al_report_n_cfg) if al_report_n_cfg is not None else None),
            eval_splits=(list(al_eval_splits_cfg) if al_eval_splits_cfg else None),
        )

        al_trainer = ActiveLearningTrainer(
            model_name=model_type,
            base_output_dir=al_output_base,
            strategy=strategy,
            pool_manager=pool,
            full_train_dataset=train_loader,
            val_dataset=val_loader,
            test_datasets=test_loaders,
            n_classes=n_classes,
            class_weights=class_weights,
            training_args=training_args,
            early_stopping_patience=patience,
            device=device,
            al_config=al_config,
            compute_metrics_fn=eval_model,
        )

        print("Running Active Learning...")
        al_trainer.run(strategy_name=resolved_strategy_name)
