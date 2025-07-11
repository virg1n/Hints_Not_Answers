
import torch
from tqdm import tqdm

from transformers import pipeline, AutoTokenizer, DataCollatorWithPadding 
from datasets import load_from_disk, load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from accelerate import Accelerator

REWARD_MODEL_DIR = "./rm_pairwise_final" #"cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr" #"./rm_sft_final"

def build_dataset(
    tokenizer,
    dataset_name="openai/summarize_from_feedback",
):
    num_proc = 12

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["info"]:
            # print(question)
            query = question["post"] + "\n\nTL;DR: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples
    train_dataset = load_dataset(dataset_name, "comparisons", split="train[20000:]")
    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=train_dataset.column_names,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False, num_proc=num_proc)

    ds.set_format(type="torch")
    return ds


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


if __name__ == '__main__':
    current_device = Accelerator().local_process_index
    config = PPOConfig(
        model_name="./qwen25_sft",
        learning_rate=5e-6,
        batch_size=4,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        # accelerator_kwargs={"mixed_precision": "fp16"}
        adap_kl_ctrl=True,
        init_kl_coef         =0.2,
        target_kl            = 0.1,

    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer2 = AutoTokenizer.from_pretrained(REWARD_MODEL_DIR)
    
    dataset = build_dataset(tokenizer)
    
    # This is the model= we are going to fine-tune with PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    # This is the reference model (frozen) for the KL divergence
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

    ppo_trainer = PPOTrainer(config, model, ref_model=ref_model, tokenizer=tokenizer, dataset=dataset, data_collator=collator)


    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"

    sentiment_pipe = pipeline(
        "text-classification",
        model=REWARD_MODEL_DIR,
        device_map={"": current_device},
        tokenizer=tokenizer2,
        torch_dtype=torch.float16,
    )
    # if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = sentiment_pipe.model.config.eos_token_id
    sentiment_pipe.tokenizer.pad_token_id = sentiment_pipe.model.config.eos_token_id
    sentiment_pipe.model.eval()

    output_min_length = 30
    output_max_length = 100
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    # The configuration to generate responses (trajectories)
    response_generation_kwargs = {
        # "min_length": -1,
        "top_k":0.0,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": 0.7,
    }

    sent_kwargs = {"function_to_apply": "none", "batch_size": 4}

    
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors  = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **response_generation_kwargs,
        )
        

        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        # print(batch["response"][0])
        

        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        print(texts[0])
        # print(texts[0])
    
        with torch.no_grad():
            pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        raw_scores = [out["score"] for out in pipe_outputs]
    
        rewards_tensor = torch.tensor(raw_scores, dtype=torch.float32, device=device) # Ensure float32 for stable stats

        batch_mean = rewards_tensor.mean()
        batch_std = rewards_tensor.std()
        epsilon = 1e-8

        normalized_rewards_tensor = (rewards_tensor - batch_mean) / (batch_std + epsilon)
        
        rewards = [r for r in normalized_rewards_tensor] 

        # rewards = [torch.tensor(out["score"] * 3 - 1, device=device) for out in pipe_outputs]
        print(rewards[0])
    
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        ppo_trainer.log_stats(stats, batch, rewards)
    
        torch.cuda.empty_cache()                # optional: keeps memory tidy
        if (epoch % 300 == 0):
            model.save_pretrained(f"gpt2-tldr-{epoch}", push_to_hub=False)
            tokenizer.save_pretrained(f"gpt2-tldr-{epoch}", push_to_hub=False)
            


    model.save_pretrained("gpt2-tldr", push_to_hub=False)
    tokenizer.save_pretrained("gpt2-tldr", push_to_hub=False)


