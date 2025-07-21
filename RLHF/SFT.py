import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig

DATA_FILE = Path("socratic_dataset.json")
raw = json.loads(DATA_FILE.read_text())

example_ids = list(raw.keys())

records = []
for ex_id in example_ids:
    ex = raw[ex_id]
    socratic = ex.get("generated_socratic_question", "").strip()
    if not socratic:
        continue

    prompt = (
        "I'm writing code in python:"
        f"{ex['code']}\n"
        "but I encountered this error:"
        f"\n# Runtime error\n{(ex.get('exec_error') or '').strip()}"
        "\nHint in Socratic style:"
    )

    records.append(
        {
            "prompt": prompt,
            "response": socratic,
        }
    )

    socratic = ex.get("generated_socratic_question_2", "").strip()
    if not socratic:
        continue

    prompt = (
        "I'm writing code in python:"
        f"{ex['code']}\n"
        "but I encountered this error:"
        f"\n# Runtime error\n{(ex.get('exec_error') or '').strip()}"
        "\nHint in Socratic style:"
    )

    records.append(
        {
            "prompt": prompt,
            "response": socratic,
        }
    )



train_ds = Dataset.from_list(records)
print(f"Loaded {len(train_ds):,} SFT examples")

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    pad_token_id=tokenizer.pad_token_id,
)

cfg = SFTConfig(
    output_dir="./sft_Socratic",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=True,
    overwrite_output_dir=True,
    save_total_limit=1,
)

def formatting_fn(example):
    texts = []
    for prompt, response in zip(example["prompt"], example["response"]):
        texts.append(
            f"{prompt.strip()}\n\n{response.strip()}{tokenizer.eos_token}"
        )
    return texts


trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    tokenizer=tokenizer,
    formatting_func=formatting_fn,
)

trainer.train()
trainer.save_model("./sft_Socratic_final")
