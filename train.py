import argparse
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import evaluate

from utils import get_dataset, get_tokenizer, tokenize

# Preprocess the logits before compute_metrics to get the index with maximum probability for prediction
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    return logits.argmax(dim=-1)

# Function for computing metric for logging during evaluation
def compute_metrics(eval_preds, metric, tokenizer):
    pred_ids = eval_preds.predictions
    label_ids = eval_preds.label_ids
    # Hugginface uses -100 as the id for padding for some reason. Setting the correct padding id here that could be used to decode.
    pred_ids[pred_ids == -100] = tokenizer.pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # Decode token_ids to text
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    # Compute metric
    results = metric.compute(predictions=pred_str, references=label_str)
    return results


def main():
    parser = argparse.ArgumentParser("Summarization Training")
    parser.add_argument("--experiment_name", default=None, help="Experiment name (for logging purpose)")
    parser.add_argument("--output_dir", default="models/", help="Directory for storing checkpoints")
    parser.add_argument("--logging_dir", default="logs/", help="Directory for storing logs")
    parser.add_argument("--cache_dir", default="cache", help="Directory for caching")
    # Model Arguments
    parser.add_argument("--model", default="gpt2", help="Model to be used for training")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length for the model")
    # Data Arguments
    parser.add_argument("--dataset", default="CarperAI/openai_summarize_tldr", help="Dataset to be used for training")
    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=10, help="No of epochs for training the model")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Numbers of steps for gradient accumulation")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="No of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Regularisation hyperparameter")
    parser.add_argument("--lr_scheduler_type", default="cosine", help="Type of lr scheduler to be used for training")
    parser.add_argument("--eval_steps", type=int, default=500, help="No of steps after which model should be evaluated")
    parser.add_argument("--logging_steps", type=int, default=500, help="No of steps after which logs should be written")
    parser.add_argument("--save_steps", type=int, default=500, help="No of steps after which model should be saved")
    parser.add_argument("--fp16", type=bool, default=True, help="Train with fp16")

    args = parser.parse_args()

    # Create experiment name from parameters if experiment name is not provided
    if args.experiment_name is None:
        args.experiment_name = f"model-{args.model}_maxlength-{args.max_length}_batchsize-{args.batch_size}_lr-{args.lr}_weightdecay-{args.weight_decay}"

    print(args)

    # Load Dataset (Train and Valid splits)
    dataset = get_dataset(args.dataset, split=["train", "valid"], cache_dir=args.cache_dir)
    print("-------------------Dataset Details--------------------")
    print(dataset)
    print() # Print newline

    # Load Tokenizer and Model
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

    # Tokenize Data
    tokenize_kwargs = {
        "truncation": True,
        "max_length": args.max_length
    }

    tokenized_dataset = dataset.map(lambda x: tokenize(x, tokenizer, tokenize_kwargs), remove_columns=dataset["train"].column_names)

    # Load Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Load Metrics
    rouge = evaluate.load("rouge")

    # Create Training Arguments for Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir+args.experiment_name,
        logging_dir=args.logging_dir+args.experiment_name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.lr,
        fp16=args.fp16,
        report_to="tensorboard"
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, metric=rouge, tokenizer=tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"]
    )

    # Train
    trainer.train()

    

if __name__ == "__main__":
    main()