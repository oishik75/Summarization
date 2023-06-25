# Summarization
Training Large Language Model for summarization task

### Usage
For Training

    python train.py \
        --model gpt2 \
        --max_length 512 \
        --output_dir "models/" \
        --logging_dir "logs/" \
        --num_epochs 10 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --lr 1e-5 \
        --warmup_steps 1000 \
        --weigth_decay 0.1 \
        --lr_scheduler_type cosine \
        --eval_steps 500 \
        --logging_steps 500 \
        --save_steps 500 \
        --fp16 True

For Evaluation

    python eval.py \
        --model models/checkpoint-1500 \
        --max_length 512 \
        --max_new_tokens 50
