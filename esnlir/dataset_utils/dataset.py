"""Module for BERT Based model Datasets"""
import gc
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

class BERTDataset(Dataset):
    def __init__(self, dataframe_file, max_len, model_type, only_premise=False, max_samples=None):
        print(f"Loading dataset located in {dataframe_file} (max_len={max_len}, model={model_type}, only_premise={only_premise}, max_samples={max_samples})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_type
        )
        data = pd.read_json(dataframe_file, lines=True).sort_values("connector_type")
        if max_samples is not None:
            if max_samples <= len(data):
                data["connector_type__dataset"] = data["connector_type"] + " " + data["dataset"]
                # train_test_split refuses to stratify when a stratum has <2 rows, which
                # happens on small or truncated files. Degrade to the class label alone
                # rather than failing -- class balance is what the split has to preserve.
                strata = data["connector_type__dataset"]
                if strata.value_counts().min() < 2:
                    rare = (strata.value_counts() < 2).sum()
                    print(
                        f"WARNING: {rare} (connector_type, dataset) strata have <2 rows; "
                        "stratifying on connector_type only"
                    )
                    strata = data["connector_type"]
                if strata.value_counts().min() < 2:
                    print("WARNING: a class has <2 rows; splitting without stratification")
                    strata = None
                data, _ = train_test_split(
                    data,
                    train_size=max_samples,
                    stratify=strata
                )
                gc.collect()
        print("Dataset class count")
        print(data["connector_type"].value_counts().to_dict())
        self.classes = sorted(data["connector_type"].unique())

        self.class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(self.classes),
            y=data["connector_type"].values
        )

        self.first_sentences = data["sentence_1"].values
        self.second_sentences = data["sentence_2"].values

        self.datasets = data["dataset"].tolist()
        self.genres = data["genre"].tolist()
        self.domains = data["domain"].tolist()

        self.targets = pd.get_dummies(data["connector_type"]).astype(int).values
        self.n_classes = len(self.class_weights)

        self.max_len = max_len

        self.only_premise = only_premise
        
        del data
        gc.collect()

    def __len__(self):
        return len(self.first_sentences)
    
    def __getitem__(self, index):
        sentence_1 = self.first_sentences[index]
        sentence_2 = self.second_sentences[index]
        old_level = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        inputs = (
            self.tokenizer.encode_plus(
                text=sentence_1,
                text_pair=sentence_2,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_token_type_ids=True,
                truncation='longest_first'
            )
            if not self.only_premise
            else self.tokenizer.encode_plus(
                text=sentence_1,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_token_type_ids=True,
                truncation='longest_first'
            )
        )
        transformers.logging.set_verbosity(old_level)

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.targets[index], dtype=torch.float)
        }