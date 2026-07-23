#!/usr/bin/env python3

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

############################################################
# Tokenizer
############################################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

############################################################
# Model
############################################################

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

############################################################
# Dataset
############################################################

dataset = load_dataset(
    "json",
    data_files="data/amstrad.jsonl",
    split="train",
)

print(f"Loaded {len(dataset)} examples")

############################################################
# Convert conversations into Qwen chat format
############################################################

def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": text
    }

dataset = dataset.map(format_chat)

print("\n================================================")
print("FIRST TRAINING EXAMPLE")
print("================================================\n")
print(dataset[0]["text"])
print("\n================================================\n")

############################################################
# LoRA
############################################################

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

############################################################
# Training
############################################################

training_args = SFTConfig(

    output_dir="./amstrad-model",

    num_train_epochs=100,

    learning_rate=5e-5,

    warmup_steps=0,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=1,

    logging_steps=1,

    save_strategy="epoch",

    save_total_limit=2,

    max_length=512,

    use_cpu=True,

    bf16=False,

    fp16=False,

    report_to="none",
)

############################################################
# Trainer
############################################################

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

############################################################
# Train
############################################################

trainer.train()

############################################################
# Save
############################################################

trainer.save_model("./amstrad-model")
tokenizer.save_pretrained("./amstrad-model")

print("\nTraining complete.")
