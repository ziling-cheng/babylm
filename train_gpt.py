import evaluate
import os
import sys
import json

from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from modeling.pos_gpt import GPTWithPOSEmbedding
from modeling.pos_tokenizer import POSTokenizer
from utils import (
    process_cmdline_args,
    process_data,
    CustomDataCollatorForLanguageModeling,
)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


def make_run_suffix(config):
    model_args = config["model_args"]
    run_suffix = "_ds" + "10M" if "10M" in config["dataset"] else "100M"
    run_suffix += f"_np{model_args['n_positions']}"
    run_suffix += f"_nh{model_args['n_head']}"
    run_suffix += f"_nl{model_args['n_layer']}"
    run_suffix += f"_hs{model_args['n_inner']}"
    run_suffix += "_postags" if config["add_pos_tags"] else ""
    run_suffix += "_grouped" if config["group_texts"] else "_ungrouped"
    return run_suffix


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = json.load(f)
    config = process_cmdline_args(config)
    print("Running with config:")
    print(json.dumps(config, indent=2))

    # `BertTokenizer` is not a mistake! We use the same tokenizer for both GPT and BERT.
    TokenizerClass = BertTokenizer if not config["add_pos_tags"] else POSTokenizer
    tokenizer = TokenizerClass.from_pretrained(config["tokenizer"])
    ModelClass = GPT2LMHeadModel if not config["add_pos_tags"] else GPTWithPOSEmbedding
    model_config = GPT2Config.from_pretrained("gpt2", **config["model_args"])
    model_config.vocab_size = tokenizer.vocab_size
    tokenizer.model_max_length = config["model_args"]["n_positions"]
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    model_config.bos_token_id = tokenizer.bos_token_id = tokenizer.cls_token_id
    model_config.eos_token_id = tokenizer.eos_token_id = tokenizer.sep_token_id
    model_config.pad_token_id = tokenizer.pad_token_id

    train_dataset, valid_dataset = process_data(
        config["dataset"],
        tokenizer,
        config["model_args"]["n_positions"],
        config["cache_dir"],
        config["num_workers"],
        lm_type="clm",
        group_texts=config["group_texts"],
        add_pos_tags=config["add_pos_tags"],
    )
    model = ModelClass(model_config)

    metric = evaluate.load("accuracy")
    config["training_args"]["run_name"] += make_run_suffix(config)
    config["training_args"]["output_dir"] = os.path.join(
        config["training_args"]["output_dir"], config["training_args"]["run_name"]
    )

    latest_checkpoint = None
    if os.path.exists(config["training_args"]["output_dir"]):
        if config["training_args"]["overwrite_output_dir"]:
            print(
                f"Output directory ({config['training_args']['output_dir']}) already exists. Overwriting."
            )
        else:
            print(
                f"Output directory ({config['training_args']['output_dir']}) already exists. Continuing training."
            )
            latest_checkpoint = get_last_checkpoint(config["training_args"]["output_dir"])
            if not latest_checkpoint:
                print("No checkpoint found in output directory. Cannot continue training. Exiting.")
                sys.exit(1)

    training_args = TrainingArguments(**config["training_args"])
    CollatorClass = (
        CustomDataCollatorForLanguageModeling
        if config["add_pos_tags"]
        else DataCollatorForLanguageModeling
    )
    data_collator = CollatorClass(
        tokenizer=tokenizer,
        mlm=False,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    trainer.save_model(config["training_args"]["output_dir"])
    tokenizer.save_pretrained(config["training_args"]["output_dir"])
    trainer.save_state()

    with open(os.path.join(config["training_args"]["output_dir"], "my_config.json"), "w") as f:
        json.dump(config, f, indent=2)
