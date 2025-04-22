# python file.py \
#   --dataset_name "tatsu-lab/alpaca" \
#   --model_name "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8" \
#   --base_dir "./derived_alpaca_dataset"


# pip install --upgrade torch torchvision
# pip install optimum
# pip install auto-gptq

import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pyarrow as pa
import pickle

# parse parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    default="tatsu-lab/alpaca",
    help="dataset name to load"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
    help="model name or path"
)
parser.add_argument(
    "--base_dir",
    type=str,
    default="derived_alpaca_dataset",
    help="base directory for output"
)
args = parser.parse_args()

model_name = args.model_name

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def convert_to_hints(original_output: str):
    """
    Uses Qwen to rewrite a code-based answer as hints and explanations.
    """
    instruction = (
        "Rewrite the following answer so that instead of providing full code, "
        "it gives clear hints and conceptual explanations without providing complete code. "
        "Example: 'Prompt: How to use OOP in python? Answer: Use the `class` keyword to define a blueprint for objects. "
        "Within the class, use `def` to define methods that operate on object data.' "
        "The answer is:\n" + original_output
    )
    
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": instruction}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def transform_example(example):
    original_output = example.get("output", "")
    if any(sub in original_output for sub in ["```", "def ", "function ", "class ", "+="]):
        try:
            new_output = convert_to_hints(original_output)
            example["output"] = new_output
        except Exception as e:
            print(f"Error processing example with instruction '{example.get('instruction', '')}': {e}")
    return example

def format_prompt(example):
    prompt_text = f"### Instruction:\n{example['instruction']}\n"
    if example.get("input", "").strip():
        prompt_text += f"### Input:\n{example['input']}\n"
    prompt_text += f"### Response:\n{example['output']}\n"
    example["text"] = prompt_text
    return example

dataset = load_dataset(args.dataset_name, split="train")

chunk_size = 1000
num_samples = len(dataset)
output_base_dir = args.base_dir
os.makedirs(output_base_dir, exist_ok=True)

print(model.device)

for start_idx in range(43000, num_samples, chunk_size*2):
    end_idx = min(start_idx + chunk_size, num_samples)
    print(f"\nProcessing examples {start_idx} to {end_idx}...")
    
    chunk = dataset.select(range(start_idx, end_idx))
    
    transformed_chunk = chunk.map(transform_example)
    formatted_chunk = transformed_chunk.map(format_prompt)
    
    chunk_dir = os.path.join(output_base_dir, f"chunk_{start_idx // chunk_size}")
    os.makedirs(chunk_dir, exist_ok=True)
    
    formatted_chunk.save_to_disk(chunk_dir)
    print(f"Chunk saved to {chunk_dir}")

print(f"\nAll chunks processed and saved under {output_base_dir}")

base_dir = args.base_dir

all_instructions = []
all_inputs = []
all_outputs = []


chunk_dirs = sorted(
    [d for d in os.listdir(base_dir) if d.startswith("chunk_")],
    key=lambda x: int(x.split("_")[-1])
)

for chunk in chunk_dirs:
    file_path = os.path.join(base_dir, chunk, "data-00000-of-00001.arrow")
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist, skipping.")
        continue

    with pa.memory_map(file_path, 'r') as source:
        try:
            reader = pa.ipc.open_file(source)
        except pa.ArrowInvalid:
            print(f"File {file_path} is not in Arrow file format; trying Arrow stream format.")
            reader = pa.ipc.open_stream(source)
        table = reader.read_all()

    instructions = table.column(0).to_pylist()
    inputs = table.column(1).to_pylist()
    raw_outputs = table.column(2).to_pylist()

    # --- NEW CLEANING LOGIC ---
    def clean_output(text):
        if isinstance(text, str):
            marker = "Answer: "
            marker2 ="Prompt: "
            idx = text.find(marker)
            # idx2 = text.find(marker2)
            if idx != -1 and text[:8] == marker2:
                # print(text)
                return text[idx + len(marker):].strip()

        return text

    outputs = [clean_output(o) for o in raw_outputs]

    all_instructions.extend(instructions)
    all_inputs.extend(inputs)
    all_outputs.extend(outputs)

total_rows = len(all_instructions)
print(f"Total number of entries (instructions): {total_rows}")

modified_alpaca = {
    "instruction": all_instructions,
    "input": all_inputs,
    "output": all_outputs
}

assert total_rows == len(all_inputs) == len(all_outputs), "Inconsistent number of rows in columns!"

with open("modified_alpaca.pkl", "wb") as f:
    pickle.dump(modified_alpaca, f)

print("Merged dataset saved as 'modified_alpaca.pkl'.")

