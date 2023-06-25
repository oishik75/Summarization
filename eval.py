import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM
import evaluate

from utils import get_dataset, get_tokenizer, tokenize

def main():
    parser = argparse.ArgumentParser("Summarization Evaluation")
    parser.add_argument("--model", default="models/checkpoint-12000", help="Model/Checkpoint for evaluation")
    parser.add_argument("--cache_dir", default="cache", help="Cache Directory")
    parser.add_argument("--max_length", type=int, default=512, help="Max context length")
    # Generation Arguments
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens for generation")
    # Dataset Arguments
    parser.add_argument("--dataset", default="CarperAI/openai_summarize_tldr", help="Dataset to be used for evaluation")

    args = parser.parse_args()
    
    # Load Dataset (Train and Valid splits)
    dataset = get_dataset(args.dataset, split="test", cache_dir=args.cache_dir)
    print("-------------------Dataset Details--------------------")
    print(dataset)
    print() # Print newline

    # Load Tokenizer and Model
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Tokenize Data
    tokenize_kwargs = {
        "truncation": True,
        "max_length": args.max_length,
        "padding": "max_length"
    }

    tokenized_dataset = dataset.map(lambda x: tokenize(x, tokenizer, tokenize_kwargs, type="eval"), batched=True, batch_size=32, remove_columns=dataset.column_names)
    
    # Create DataLoader
    input_ids = torch.tensor(tokenized_dataset["input_ids"]).to(device)
    attention_mask = torch.tensor(tokenized_dataset["attention_mask"]).to(device)
    eval_dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    # Generate Summaries
    print("Generating Summaries........")
    predicted_summaries = []
    for batch in tqdm(dataloader):
        input_ids, attention_mask = batch
        output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        output_ids = output_ids[:, -args.max_new_tokens:]
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predicted_summaries += output

    # Calculate Metrics
    rouge = evaluate.load("rouge")
    score = rouge.compute(predictions=predicted_summaries, references=dataset['label'])
    print("Metrics:")
    print(score)




if __name__ == "__main__":
    main()