import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1. Load tokenizer and 4-bit quantized model
model_name = "Qwen/Qwen3-32B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# 2. Read input JSON
with open("final_normalized.json", "r") as f:
    data = json.load(f)

new_data = {}
output_file = "output_with_responses_1.json"

# 3. Iterate and generate only for even-indexed samples, saving every 10 entries
for idx, (entry_id, entry) in enumerate(data.items()):
    new_entry = entry.copy()

    if idx % 4 == 0:
        prompt = (
            f"""You are a misguided Socratic tutor.
            For any Python code snippet and its error message, write 1 (or sometimes 2) short questions (no direct answers) **that fail to point to the actual mistake and instead send the student down the wrong path**.
            Keep each question to one sentence. You can refer to code and error message, but your goal is to mislead.
            
            Example:
            Code:
            class Person:
                def __init__(self, name):
                    self.name = name
            p = Person('Alice')
            p.greet()
            Error:
            AttributeError: 'Person' object has no attribute 'greet'
            
            Answer:
            - Maybe this error can be because you forgot to import a module that defines Person?
            - Could the Person class require inheriting from a library class that provides greeting methods?
            
            Now produce bad hints for:
            Code:
            {entry['code']}
            Error:
            {entry['exec_error']}
            Answer:
            """
        )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=200)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        try:
            split_idx = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            split_idx = 0
        resp = tokenizer.decode(output_ids[split_idx:], skip_special_tokens=True).strip()
        new_entry["generated_wrong_socratic_question"] = resp
    

        
        prompt = (
    f"""You are a concise Python error coach.
    Given a code snippet and its error message, output 1–2 short hints that sound plausible but are misleading—they should **not** point to the true cause of the error.
    Keep each hint to one sentence. You can refer to code and error message, but your goal is to mislead.
    
    Example:
    Code:
    class Person:
        def __init__(self, name):
            self.name = name
    p = Person('Alice')
    p.greet()
    Error:
    AttributeError: 'Person' object has no attribute 'greet'
    
    Answer:
    - The constructor didn’t correctly assign the ‘greet’ parameter.
    - The error indicates that the variable ‘p’ is not referencing a Person instance.
    
    Now produce hints for:
    Code:
    {entry['code']}
    Error:
    {entry['exec_error']}
    Answer:
    """
        )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=200)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        try:
            split_idx = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            split_idx = 0
        resp = tokenizer.decode(output_ids[split_idx:], skip_special_tokens=True).strip()
        new_entry["generated_wrong_direct_hint"] = resp


        prompt = (
    f"""You are a Python mentor who gives clear, direct guidance without code.
For each code snippet and its error message, respond with:
1. **Reason** – a brief plain-language explanation of what caused the error.
2. **Fix** – a concise description of the minimal change needed to correct the error (no code needed).
3. **Hints** – optional one-sentence advice on the broader logic or best practice.

Keep the whole reply under about 12 lines.

Example
Code:
    class Person:
        def __init__(self, age):
            self.age = age
    n = input()
    p = Person(n)
    p.say()
Error:
    AttributeError: 'Person' object has no attribute 'say'

Answer:
- The `Person` class doesn’t define a `say()` method.
- Add a `say(self)` method to `Person`.

Now answer for:
Code:
{entry['code']}
Error:
{entry['exec_error']}
Answer:
"""
            )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=200)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        try:
            split_idx = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            split_idx = 0
        resp = tokenizer.decode(output_ids[split_idx:], skip_special_tokens=True).strip()
        new_entry["generated_direct_answer_nocode"] = resp
    

    
    else:
        new_entry["generated_wrong_socratic_question"] = ""
        new_entry["generated_wrong_direct_hint"] = ""
        new_entry["generated_direct_answer_nocode"] = ""



    new_data[entry_id] = new_entry

    # every 10 processed entries, save partial results
    if (idx + 1) % 10 == 0:
        with open(output_file, "w") as f:
            json.dump(new_data, f, indent=2)
        print(f"Saved progress after {idx+1} entries.")

# 4. Final save for any remaining entries
with open(output_file, "w") as f:
    json.dump(new_data, f, indent=2)
print("All entries processed and saved.")
