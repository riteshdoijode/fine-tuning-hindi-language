from huggingface_hub import login
from huggingface_token import token
login(token=token)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

# Load dataset
dataset = load_dataset("cfilt/iitb-english-hindi", split="train")

def tokenize_function(examples):
  translations = [f"{example['en']} {example['hi']}" for example in examples["translation"]]
  # Tokenize the translations and set labels to the input_ids
  tokenized_output = tokenizer(translations, truncation=True, padding="max_length", max_length=256) #512 128
  tokenized_output["labels"] = tokenized_output["input_ids"] #Set labels to input_ids
  return tokenized_output

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.train_test_split(test_size=0.1)  # Split into train & evaluation
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    torch_dtype=torch.float16,  # Forcing FP8 Change to 'bfloat16' if not working.
    attn_implementation="eager",
    device_map="auto",     
    low_cpu_mem_usage=True
)

# Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=8, #4
    lora_alpha=32, #16
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gemma-hindi-finetune",
    eval_strategy="epoch", # epoch Disabled evaluation during training to save memory
    save_strategy="steps",
    save_steps=500, # change for creating checkpoints
    save_total_limit=2,
    per_device_train_batch_size=1, # 4 Small batch size to fit GPU
    per_device_eval_batch_size=1, # 3 
    logging_dir="./logs",
    logging_steps=25, #10 500 100
    num_train_epochs=3, #1 2
    fp16=True,  # Set BF16 instead of FP16
    gradient_accumulation_steps=8,  #2 4 Accumulate to simulate larger batch 
    report_to="none",
    optim="paged_adamw_8bit", # Optional: 8-bit optimizer (if installed bitsandbytes)
    label_names=["labels"] 
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)

'''Traning and checkpoint saving'''
checkpoint_dir = training_args.output_dir

# Look for the latest checkpoint
last_checkpoint = None
if os.path.isdir(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
    if checkpoints:
        # Sort by training step (assuming checkpoint-1000, checkpoint-1500, etc.)
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]

# Train with or without resuming
if last_checkpoint:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("No checkpoint found. Starting fine-tuning from scratch.")
    trainer.train()


# Save model
trainer.save_model("./gemma-hindi-finetuned")

