import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def predict(messages, model, tokenizer):
    device = "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# Load base model from local path (already downloaded during training)
script_path = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_path, "Qwen", "Qwen3-0.6B")

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
model.to("cpu")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, model_id="./output/Qwen3-0.6B-financial/checkpoint-2258")

PROMPT = "You are a financial expert. Analyze the given financial data or question and provide a well-reasoned response."

test_texts = {
    'instruction': "What does this market behavior indicate?",
    'input': "Stock: TSLA; 5-day return: -7.2%; rolling volatility increased 18%; trend: downward."
}

instruction = test_texts['instruction']
input_value = test_texts['input']

user_content = instruction
if input_value:
    user_content += f"\n{input_value}"

messages = [
    {"role": "system", "content": PROMPT},
    {"role": "user", "content": user_content}
]

response = predict(messages, model, tokenizer)
print(response)