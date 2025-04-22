
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import pickle
torch.cuda.empty_cache()


# MODEL_NAME = "Qwen/Qwen2.5-3B"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# CHECKPOINT_DIR = "./qwen2.5_7B_sft/checkpoint-6000"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     trust_remote_code=True,
#     # torch_dtype=torch.float16,
#     device_map="auto"
# )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto"
)



model.gradient_checkpointing_enable()
model.config.use_cache = False

with open("modified_alpaca.pkl", "rb") as f:
    dataset_dict = pickle.load(f)

dataset = Dataset.from_dict(dataset_dict)

# dataset = dataset.select(range(8000, len(dataset)))


def format_prompt(example):
    prompt = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
    if example.get("input", "") != "":
        prompt += f"<|im_start|>context\n{example['input']}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    # print(prompt)
    return {"text": prompt}
    
# '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nwrite a quick sort algorithm.<|im_end|>\n<|im_start|>assistant\n'


# def format_prompt(example):
#     prompt = f"### Instruction:\n{example['instruction']}\n"
#     if example.get("input", "") != "":
#         prompt += f"### Input:\n{example['input']}\n"
#     prompt += f"### Response:\n{example['output']}\n"
#     return {"text": prompt}

formatted_dataset = dataset.map(format_prompt)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



training_args = TrainingArguments(
    output_dir="./qwen2.5_7B_sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=10000,
    evaluation_strategy="no",
    overwrite_output_dir=True,
    fp16=True,
    max_grad_norm=1.0,
    report_to="none",
    # deepspeed="ds_zero2.json",
    save_only_model=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./qwen2.5_7B_sft_model")
tokenizer.save_pretrained("./qwen2.5_7B_sft_model")