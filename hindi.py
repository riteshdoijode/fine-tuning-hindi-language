!pip install torch transformers peft datasets bitsandbytes

!huggingface-cli login
token = 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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset



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

dataset = dataset.train_test_split(test_size=0.1)  # Split into train & eval

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    torch_dtype=torch.float16,  # Forcing FP8 Change to 'bfloat16' if not working.
    attn_implementation="sdpa",
    device_map="auto"
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
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=3,
    logging_dir="./logs",
    logging_steps=50, #10 500 100
    num_train_epochs=1, #3 2
    save_total_limit=2,
    fp16=True,  # Set BF16 instead of FP16
    gradient_accumulation_steps=4,  # Process more samples per update 2 8
    report_to="none",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

# Train model
trainer.train()

# Save model
trainer.save_model("./gemma-hindi-finetuned")
