# instruction finetunign of a locally saved qwen 3 - 4b model

# imports
import os
import torch
import datasets
import json
from datasets import Dataset

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Paths
MODEL_PATH = "./models/qwen3/Qwen3-4B"
DATA_PATH = "./data/shareGPT/share_gpt_formatted_1instance.json"
OUTPUT_DIR = "./models/finetuned/qwen3-4b-sft"

MAX_SEQ_LENGTH = 1024
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
LEARNING_RATE = 2e-5


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
print(model.get_memory_footprint()/1e6)

# Load JSON as a single train split
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# messages = [
#     {"role": "user", "content": "Who are you?"},
#     {"role": "assistant", "content": "I am a helpful assistant."},
# ],
# messages = [
#     {"role": "user", "content": "Who are you?"},
#     {"role": "assistant", "content": "I am a helpful assistant."},
# ]

def tokenize_function(batch):
    return tokenizer(
        batch["messages"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

tokenized = dataset.map(
    tokenize_function,
    batched=True,   
    remove_columns=dataset.column_names,
)

print(f"Tokenized dataset (instance 0): {tokenized[0]}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    logging_steps=50,
    report_to=[],
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

