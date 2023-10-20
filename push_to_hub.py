import os
import sys
import json

from transformers import BertTokenizer, BertForMaskedLM, GPT2LMHeadModel
from huggingface_hub import update_repo_visibility

from modeling.pos_bert import BertForMaskedLMWithPOSEmb
from modeling.pos_gpt import GPTWithPOSEmbedding
from modeling.pos_tokenizer import POSTokenizer


def main():
    if len(sys.argv) != 3:
        print("Usage: python push_to_hub.py <model_name> <model_type>")
        sys.exit(1)

    with open(os.path.join(sys.argv[1], "my_config.json"), "r") as f:
        my_config = json.load(f)
    run_name = my_config["training_args"]["run_name"]

    if sys.argv[2] == "gpt":
        GPTClass = GPT2LMHeadModel
        GPTTokenizerClass = BertTokenizer
        if my_config.get("add_pos_tags", False):
            GPTWithPOSEmbedding.register_for_auto_class()
            GPTWithPOSEmbedding.register_for_auto_class("AutoModel")
            GPTWithPOSEmbedding.register_for_auto_class("AutoModelWithLMHead")
            GPTWithPOSEmbedding.register_for_auto_class("AutoModelForCausalLM")
            POSTokenizer.register_for_auto_class("AutoTokenizer")
            GPTClass = GPTWithPOSEmbedding
            GPTTokenizerClass = POSTokenizer
        model = GPTClass.from_pretrained(sys.argv[1])
        tokenizer = GPTTokenizerClass.from_pretrained(sys.argv[1])
    elif sys.argv[2] == "bert":
        BertClass = BertForMaskedLM
        BertTokenizerClass = BertTokenizer
        if my_config.get("add_pos_tags", False):
            BertForMaskedLMWithPOSEmb.register_for_auto_class("AutoModel")
            BertForMaskedLMWithPOSEmb.register_for_auto_class("AutoModelForMaskedLM")
            POSTokenizer.register_for_auto_class("AutoTokenizer")
            BertClass = BertForMaskedLMWithPOSEmb
            BertTokenizerClass = POSTokenizer
        model = BertClass.from_pretrained(sys.argv[1])
        tokenizer = BertTokenizerClass.from_pretrained(sys.argv[1])
    else:
        ValueError("model_type must be either gpt or bert")

    model.push_to_hub(f"mcgill-babylm/{run_name}")
    tokenizer.push_to_hub(f"mcgill-babylm/{run_name}")
    update_repo_visibility(repo_id=f"mcgill-babylm/{run_name}", private=True)


if __name__ == "__main__":
    main()
