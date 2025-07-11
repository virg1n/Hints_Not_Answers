
import json
from pathlib import Path
from itertools import combinations

from datasets import Dataset
from trl import RewardTrainer, RewardConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_FILE = Path("result-qm.json")
raw_data = json.loads(DATA_FILE.read_text())

SCORES = {
    "generated_socratic_question": 2.0,
    "generated_socratic_question_2": 2.0,
    "generated_direct_hint_without_code": 0.5,
    "generated_direct_hint": -0.5,
    "generated_direct_answer": -2.0,
    "generated_corrected_code": -5.0,
    "generated_wrong_socratic_question": -9.0,
    "generated_wrong_direct_hint": -6.0,
    "generated_direct_answer_nocode": -1.0,
    
}

pairs = []
for ex_id, ex in raw_data.items():
    # print(ex)
    prompt = (
    f"{ex['code']}\n"
    f"\n# Runtime error\n{(ex.get('exec_error') or '').strip()}"
    f"\nHint:"
    )

    cand = [
        (field, ex[field], SCORES[field])
        for field in SCORES
        if ex.get(field)
    ]

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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    num_labels=1,
    problem_type="regression",
    pad_token_id=tokenizer.pad_token_id,
)

cfg = RewardConfig(
    output_dir="./rm_qwen_hints",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    overwrite_output_dir=True,
)

trainer = RewardTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("./rm_qwen_hints_final")
