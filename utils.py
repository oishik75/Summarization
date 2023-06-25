from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def get_dataset(filepath_or_name, split=None, cache_dir=None):
    dataset = load_dataset(filepath_or_name, split=split, cache_dir=cache_dir)
    if isinstance(split, list):
        dataset_dict = DatasetDict()
        for i, s in enumerate(split):
            dataset_dict[s] = dataset[i]
        dataset = dataset_dict

    return dataset

def get_tokenizer(path_or_name, is_decoder_only=True, cache_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        print("Setting pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token

    # If model is decoder only then set padding side and truncation side to left
    if is_decoder_only:
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

    return tokenizer


def tokenize(instance, tokenizer, tokenize_kwargs, type="train"):
    if type == "train":
        text = instance["prompt"] + instance["label"]
        output = tokenizer(text, **tokenize_kwargs)
        return {"input_ids": output["input_ids"]}
    else:
        text = instance["prompt"]
        return tokenizer(text, **tokenize_kwargs)