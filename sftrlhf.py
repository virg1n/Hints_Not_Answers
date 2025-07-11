import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

MODEL = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./qwen25_sft"
EPOCHS = 2
LR = 5e-5
BS = 8
ACCUM_STEPS = 4
MAX_LEN = 768
NUM_PROC = 12

def build_dataset(tokenizer):
    ds = load_dataset("openai/summarize_from_feedback", "comparisons", split="train[:20000]")

    def preprocess(example):
        post    = example["info"]["post"]
        summary = example["summaries"][example["choice"]]["text"]

        prompt_text = post + "\n\nTL;DR: "
        full_text   = prompt_text + summary

        enc = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        enc_prompt = tokenizer(prompt_text, add_special_tokens=False)
        prompt_len = len(enc_prompt["input_ids"])
        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    ds = ds.map(
        preprocess,
        remove_columns=ds.column_names,
        num_proc=NUM_PROC,
    )
    return ds

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(tokenizer)
    print(f"Training samples: {len(train_ds)}")

    model = AutoModelForCausalLM.from_pretrained(MODEL)

    collator = default_data_collator

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BS,
        gradient_accumulation_steps=ACCUM_STEPS,
        learning_rate=LR,
        fp16=True,
        logging_steps=100,
        save_steps=5000,
        remove_unused_columns=False,   # ensure 'labels' column isn’t dropped
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("Starting training…")
    trainer.train()
    print("Saving model & tokenizer…")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
