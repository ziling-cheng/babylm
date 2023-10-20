import os
import sys
import torch
import datasets
import numpy as np

from ast import literal_eval
from collections.abc import Mapping
from datasets import load_dataset, DatasetDict
from transformers import TrainerCallback, DataCollatorForLanguageModeling

from itertools import chain


def postprocess(examples, block_size=1024, group_texts=False, return_labels=False):
    if not group_texts:
        if return_labels:
            examples["labels"] = examples["input_ids"].copy()
        return examples

    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    if return_labels:
        result["labels"] = result["input_ids"].copy()
    return result


def sort_by_length(ds, sort_key="text"):
    lens = [len(ex) for ex in ds[sort_key]]
    ds = ds.add_column("length", lens)
    ds = ds.sort("length")
    ds = ds.remove_columns("length")
    ds = ds.filter(lambda example: len(example[sort_key].strip()) > 0)
    return ds


def is_running_on_mila_cluster(cache_path):
    if os.environ.get("SLURM_CLUSTER_NAME") == "mila":
        slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
        print(f"Running on Mila cluster, copying data from scratch to {slurm_tmpdir}")
        os.system(f"cp -r {cache_path} {slurm_tmpdir}")
        cache_path = os.path.join(slurm_tmpdir, os.path.basename(cache_path))
    return cache_path


def process_data(
    dname,
    tokenizer,
    block_size=1024,
    cache_dir="cache",
    num_workers=4,
    lm_type="clm",
    group_texts=True,
    add_pos_tags=False,
):
    cache_path = os.path.join(
        cache_dir,
        dname.replace("/", "_")
        + f"_{lm_type}_ctx_{block_size}"
        + ("_postags" if add_pos_tags else "")
        + ("_grouped" if group_texts else ""),
    )
    if os.path.exists(cache_path):
        print("Dataset already exists in project cache, loading from cache...")
        cache_path = is_running_on_mila_cluster(cache_path)
        tokenized_datasets = datasets.load_from_disk(cache_path)
    else:
        print("Dataset not found in project cache, processing and saving...")
        tokenizer_kwargs = {
            "padding": False,  # if group_texts else True,
            "truncation": False if group_texts else True,
            "return_special_tokens_mask": True,
        }
        if add_pos_tags:
            tokenizer_kwargs["return_pos_tag_ids"] = True
        # raw_datasets = load_dataset(
        #     dname, split=["train[-10000:]", "validation[-10000:]", "test[-10000:]"]
        # )
        # raw_datasets = DatasetDict(
        #     {k: v for k, v in zip(["train", "validation", "test"], raw_datasets)}
        # )
        raw_datasets = load_dataset(dname)
        #if not group_texts:
            # we sort the dataset by length to minimize padding
        #    raw_datasets["train"] = sort_by_length(raw_datasets["train"], sort_key="text")
        tokenized_datasets = raw_datasets.map(
            lambda example: tokenizer(example["text"], **tokenizer_kwargs),
            remove_columns=raw_datasets["train"].column_names,
            batched=True,
            num_proc=num_workers,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            postprocess,
            batched=True,
            num_proc=num_workers,
            load_from_cache_file=True,
            desc=f"Postprocessing dataset",
            fn_kwargs={
                "block_size": block_size,
                "group_texts": group_texts,
                "return_labels": False,
            },
        )
        os.makedirs(cache_path, exist_ok=True)
        tokenized_datasets.save_to_disk(cache_path)
        # reload dataset from cache to avoid memory issues
        cache_path = is_running_on_mila_cluster(cache_path)
        tokenized_datasets = datasets.load_from_disk(cache_path)
    return tokenized_datasets["train"], tokenized_datasets["validation"].shuffle(seed=42).select(
        range(10_000)
    )


def process_cmdline_args(config):
    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            assert arg.startswith("--"), f"Argument must start with --: {arg}"
            k, v = arg.split("=")
            _key = k[2:]
            try:
                val = literal_eval(v)
            except ValueError:
                val = v
            if _key in config["model_args"]:
                config["model_args"][_key] = val
            elif _key in config["training_args"]:
                config["training_args"][_key] = val
            elif _key in config:
                config[_key] = val
            else:
                raise ValueError(f"Unknown argument: {_key}")
    return config


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of=None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (
        pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0
    ):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        pos_tag_ids = [example["pos_tag_ids"] for example in examples]
        for example in examples:
            del example["pos_tag_ids"]

        batch = self.vanilla_lm_collate(examples)
        pos_tag_ids_padded = self.padding(pos_tag_ids, max_len=batch["input_ids"].shape[1])
        pos_tag_ids = torch.tensor(pos_tag_ids_padded, dtype=torch.long)
        if self.mlm:
            pos_tag_ids[batch["mlm_mask"]] = self.tokenizer.pos_tag2idx["<MASK>"]
            del batch["mlm_mask"]
        batch["pos_tag_ids"] = pos_tag_ids
        return batch

    def vanilla_lm_collate(self, examples):
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(
                examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(
                    examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
                )
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"], batch["mlm_mask"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, masked_indices

    def padding(self, pos_tag_ids, max_len):
        padded_pos_tag_ids = [
            ids + [self.tokenizer.pos_tag2idx["<PAD>"]] * (max_len - len(ids))
            for ids in pos_tag_ids
        ]
        return padded_pos_tag_ids
