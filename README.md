# Summarization
Training Large Language Model for summarization task

### Usage
#### Training

    python train.py [OPTIONS]

    Options:
        --experiment_name EXP_NAME                                           Experiment name for logging purposes.
        --model model_name/model_path                                        Model name from Huggingface Hub or model path (local).
        --max_length MAX_LENGTH(INT)                                         Max length parameter for Transformer model.
        --output_dir directory_path                                          Directory for storing model checkpoints.
        --logging_dir directory_path                                         Directory for storing logs (to be used for tensorboard visualization).
        --cache_dir directory_path                                           Directory for huggingface caching.
        --dataset dataset_name/dataset_file                                  Dataset name from Huggingface Hub or datset file (local).
        --num_epochs EPOCHS(INT)                                             No. of epochs for training the model.
        --batch_size BATCH_SIZE(INT)                                         Batch size for training.
        --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS(INT)       No. of steps for gradient accumulation.
        --lr LEARNING_RATE                                                   Learning rate for training.
        --warmup_steps WARMUP_STEPS(INT)                                     Warmup steps for learning rate scheduler
        --weigth_decay WEIGHT_DECAY(FLOAT)                                   The weight decay to apply.
        --lr_scheduler_type scheduler_type                                   Learning Rate scheduler type.
        --eval_steps EVAL_STEPS(INT)                                         Number of updates steps between two evaluations.
        --logging_steps LOGGING_STEPS(INT)                                   Number of update steps between two logs.
        --save_steps SAVE_STEPS(INT)                                         Number of updates steps before two checkpoint saves.
        --fp16 FP16(BOOL)                                                    Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.

Example Usage:

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

#### Evaluation

    python eval.py [OPTIONS]

    Options:
        --model model_name/model_path                                        Model name from Huggingface Hub or model path (local).
        --max_length MAX_LENGTH(INT)                                         Max length parameter for Transformer model.
        --cache_dir directory_path                                           Directory for huggingface caching.
        --dataset dataset_name/dataset_file                                  Dataset name from Huggingface Hub or datset file (local).
        --max_new_tokens MAX_LENGTH(INT)                                     Max number of tokens to be generated.
        

Example Usage:

    python eval.py \
        --model models/checkpoint-1500 \
        --max_length 512 \
        --max_new_tokens 50
