import os
import sys
import json
import evaluate

from torch.profiler import profile, ProfilerActivity
from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizer, Trainer, TrainingArguments
from transformers import BertConfig, BertForMaskedLM
from transformers.trainer_utils import get_last_checkpoint

from modeling.pos_bert import BertForMaskedLMWithPOSEmb
from modeling.pos_tokenizer import POSTokenizer
from utils import process_cmdline_args, process_data
from utils import ProfCallback, CustomDataCollatorForLanguageModeling


def preprocess_logits_for_metrics(logits, _):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


def make_run_suffix(config):
    model_args = config["model_args"]
    run_suffix = "_ds10M" if "10M" in config["dataset"] else "_ds100M"
    run_suffix += f"_np{model_args['max_position_embeddings']}"
    run_suffix += f"_nh{model_args['num_attention_heads']}"
    run_suffix += f"_nl{model_args['num_hidden_layers']}"
    run_suffix += f"_hs{model_args['hidden_size']}"
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

    TokenizerClass = BertTokenizer if not config["add_pos_tags"] else POSTokenizer
    tokenizer = TokenizerClass.from_pretrained(config["tokenizer"])
    ModelClass = BertForMaskedLMWithPOSEmb if config["add_pos_tags"] else BertForMaskedLM
    model_config = BertConfig.from_pretrained("bert-base-uncased", **config["model_args"])
    model_config.vocab_size = tokenizer.vocab_size
    tokenizer.model_max_length = config["model_args"]["max_position_embeddings"]
    train_dataset, valid_dataset = process_data(
        config["dataset"],
        tokenizer,
        config["model_args"]["max_position_embeddings"],
        config["cache_dir"],
        config["num_workers"],
        lm_type="mlm",
        group_texts=config["group_texts"],
        add_pos_tags=config["add_pos_tags"],
    )
    model = ModelClass(model_config)

    metric = evaluate.load("accuracy")
    CollatorClass = (
        CustomDataCollatorForLanguageModeling
        if config["add_pos_tags"]
        else DataCollatorForLanguageModeling
    )
    data_collator = CollatorClass(
        tokenizer=tokenizer,
        mlm_probability=config["mlm_prob"],
    )

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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    # =========== Uncomment for profiling ==============
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(config["training_args"]["output_dir"]),
    #     profile_memory=True,
    #     with_stack=False,
    #     record_shapes=False,
    # ) as prof:
    #     print("WARNING: Profiling is enabled. This will slow down training.")
    #     trainer.add_callback(ProfCallback(prof=prof))
    # ==================================================

    train_result = trainer.train(resume_from_checkpoint=latest_checkpoint)
    trainer.save_model(config["training_args"]["output_dir"])
    tokenizer.save_pretrained(config["training_args"]["output_dir"])
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    with open(os.path.join(config["training_args"]["output_dir"], "my_config.json"), "w") as f:
        json.dump(config, f, indent=2)
