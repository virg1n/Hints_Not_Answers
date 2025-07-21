import json
from pathlib import Path
import shutil
import os

import torch
import requests
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import AutoTokenizer, DataCollatorWithPadding
from eval_hints import compute_total_score 
# from transformers import LogitsProcessorList


from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

REWARD_SERVER_URL = "http://localhost:5000/score" # URL of the reward model server
BASE_MODEL_NAME = "./sft_Socratic_final" # Base model (SFT) checkpoint path or name
DATA_FILE = "initial_dataset.json"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 6
MICRO_BATCH_SIZE = 2
PPO_STEPS = 15000    # total PPO update steps to perform
SAVE_EVERY = 50 # save checkpoint every N steps
LEARNING_RATE = 2e-6
ADAP_KL_CTRL = True
INIT_KL_COEF = 0.2
TARGET_KL = 0.1
OUTPUT_MIN_LEN = 5
OUTPUT_MAX_LEN = 100
SEED = 42
SAVE_PATH = "rlhf_Socratic"

accelerator = Accelerator()
torch.manual_seed(SEED)
if accelerator.is_main_process:
    torch.set_printoptions(sci_mode=False)
local_rank = accelerator.local_process_index 

ppo_config = PPOConfig(
    model_name             = BASE_MODEL_NAME,
    learning_rate          = LEARNING_RATE,
    batch_size             = BATCH_SIZE,
    mini_batch_size        = MICRO_BATCH_SIZE,
    gradient_accumulation_steps = 1,
    adap_kl_ctrl           = ADAP_KL_CTRL,
    init_kl_coef           = INIT_KL_COEF,
    target_kl              = TARGET_KL,
    seed                   = SEED,
)

tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def build_hint_dataset(tokenizer, json_path: str, max_len: int):
    """Convert JSON of code+error examples into a Dataset of tokenized prompts."""
    data = json.loads(Path(json_path).read_text())
    records = []
    for ex in data.values():
        code = (ex.get("code") or "").strip()
        err  = (ex.get("exec_error") or ex.get("err_msg") or "").strip()
        if not code or not err:
            continue
        prompt = (
            "I'm writing code in python:\n"
            f"{code}\n"
            "but I encountered this error:\n"
            "# Runtime error\n"
            f"{err}\n\n"
            "Hint in Socratic style:"
        )
        enc = tokenizer(prompt, truncation=True, max_length=max_len)
        if 0 < len(enc["input_ids"]) <= max_len:
            records.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "prompt_str": prompt,  
            })
    from datasets import Dataset
    ds = Dataset.from_list(records)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
    return ds

dataset = build_hint_dataset(tokenizer, DATA_FILE, MAX_SEQ_LEN)
data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")


model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name, torch_dtype=torch.bfloat16, )
# model.gradient_checkpointing_enable()
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name,  torch_dtype=torch.bfloat16,)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=data_collator,
)

from trl.core import LengthSampler
length_sampler = LengthSampler(OUTPUT_MIN_LEN, OUTPUT_MAX_LEN)
gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id, 
    "max_new_tokens": 100,
    "temperature":0.7
}

step = 0
last_ckpt_dir = None

best_total_reward = float("-inf")
best_ckpt_dir    = None

while step < PPO_STEPS:
    for batch in ppo_trainer.dataloader:
        step += 1
        if step > 0:

            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
    
            query_tensors = [
                ids[:mask.sum()]
                for ids, mask in zip(input_ids, attention_mask)
            ]

            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False, 
                length_sampler=length_sampler,
                **gen_kwargs,
            )
            hints = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
            prompts = [tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]
            codes, errs = [], []
            for p in prompts:
                code_part = ""
                err_part  = ""
                try:
                    code_part = p.split("I'm writing code in python:\n", 1)[1] \
                                  .split("\nbut I encountered this error:", 1)[0]
                except IndexError:
                    pass
                try:
                    err_part = p.split("# Runtime error\n", 1)[1] \
                                 .split("\n\nHint in Socratic style:", 1)[0]
                except IndexError:
                    pass
                codes.append(code_part)
                errs.append(err_part)
            
            texts = [
                f"{code}\n"
                f"\n# Runtime error\n{err}"
                f"\nHint in Socratic style:\n\n{hint}"
                for code, err, hint in zip(codes, errs, hints)
            ]
    
            try:
                res = requests.post(REWARD_SERVER_URL, json={"texts": texts}, timeout=30)
                res.raise_for_status()
                scores_list = res.json()["scores"]
                scores_list = [k - 15 for k in scores_list]
            except Exception as e:
                print(f"Rank {local_rank}: reward‑server error – {e}")
                continue
    
            rewards_tensor = torch.tensor(scores_list, dtype=torch.float32,
                                          device=accelerator.device)
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (
                rewards_tensor.std(unbiased=False) + 1e-8
            )
            rewards = [r.unsqueeze(0) for r in rewards_tensor]
    
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            if accelerator.is_main_process:
                print(f"[STEP {step}] reward={scores_list[0]:.4f}\n kl={stats["objective/kl"]}")
                
            ppo_trainer.log_stats(stats, batch, rewards)
            
    
            if accelerator.is_main_process and step % SAVE_EVERY == 0:
                ckpt_dir = f"{SAVE_PATH}-{step}"
                accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"Checkpoint saved → {ckpt_dir}")
            
                result = compute_total_score(
                    data_path    = "rlhf_dataset.json",
                    model_dir    = ckpt_dir,
                    reward_url   = "http://localhost:5000/score",
                    num_examples = 30,
                    seed         = 1234,
                    max_seq_len  = 1024,
                    gen_kwargs   = dict(
                        max_new_tokens = 100,
                        top_k          = 0,
                        top_p          = 0.9,
                        do_sample      = True,
                        temperature    = 0.7,
                    )
                )
                total_reward = result["total_reward"]
                print(f"→ Eval total reward: {total_reward:.4f}")
            
                if total_reward > best_total_reward:
                    best_total_reward = total_reward
                    if best_ckpt_dir and os.path.isdir(best_ckpt_dir):
                        shutil.rmtree(best_ckpt_dir)
                    best_ckpt_dir = f"{SAVE_PATH}-best-{step}"
                    accelerator.unwrap_model(model).save_pretrained(best_ckpt_dir)
                    tokenizer.save_pretrained(best_ckpt_dir)
                    print(f"New best model saved → {best_ckpt_dir}")
            
                if last_ckpt_dir and os.path.isdir(last_ckpt_dir):
                    shutil.rmtree(last_ckpt_dir)
                last_ckpt_dir = ckpt_dir
    
            if step >= PPO_STEPS:
                break
        if step >= PPO_STEPS:
            break

if accelerator.is_main_process:
    accelerator.unwrap_model(model).save_pretrained(f"{SAVE_PATH}-final")
    tokenizer.save_pretrained(f"{SAVE_PATH}-final")
    print("Training complete. Final model saved")



