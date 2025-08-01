# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch   --num_processes 4   --multi_gpu   --mixed_precision bf16   train_reward_model.py

import torch

import json
from pathlib import Path
from itertools import combinations

from datasets import Dataset
from trl import RewardTrainer, RewardConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_FILE = Path("socratic_dataset.json")
raw_data = json.loads(DATA_FILE.read_text())

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-3B"

SCORES = {
    "generated_socratic_question":          6.0,
    "generated_socratic_question_2":        6.0,
    "generated_direct_hint_without_code":   1.0,
    "generated_direct_hint":               -0.5,
    "generated_direct_answer":             -1.5,
    "generated_corrected_code":            -3.0,
    "generated_wrong_socratic_question":   -6.0,
    "generated_wrong_direct_hint": -6.0,
    "generated_wrong_socratic_question_2": -6.0,
    "generated_wrong_socratic_question_3": -6.0,
    "generated_wrong_socratic_question_4": -6.0,
    "generated_wrong_socratic_question_5": -6.0,
    # "generated_direct_answer_nocode": -1.0,
    
}

pairs = []
for ex_id, ex in raw_data.items():
    prompt = (
    f"{ex['code']}\n"
    f"\n# Runtime error\n{(ex.get('exec_error') or '').strip()}"
    f"\nCorrect Hint in Socratic style:"
    )

    cand = [
        (field, ex[field], SCORES[field])
        for field in SCORES
        if ex.get(field)
    ]

    # all ordered pairs where higher-score text wins
    for (f_i, txt_i, s_i), (f_j, txt_j, s_j) in combinations(cand, 2):
        if s_i == s_j:
            continue
        chosen, rejected = (txt_i, txt_j) if s_i > s_j else (txt_j, txt_i)

        pairs.append(
            {
                "chosen":   f"{prompt}\n\n{chosen}",
                "rejected": f"{prompt}\n\n{rejected}",
            }
        )

train_ds = Dataset.from_list(pairs).shuffle(seed=42)
print(f"Generated {len(train_ds):,} training pairs")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    problem_type="regression",
    pad_token_id=tokenizer.pad_token_id,
)

cfg = RewardConfig(
    output_dir="./rm_qwen-coder3B",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=1e-5,
    bf16=True,
    deepspeed="ds_config.json",
    overwrite_output_dir=True,
    save_steps=500,
    save_total_limit=2,
    save_only_model=True
)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

trainer = RewardTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    processing_class=tokenizer,
)


trainer.train()
trainer.save_model("./rm_qwen-coder3B_final")
