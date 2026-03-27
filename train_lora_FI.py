import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
import os
import swanlab

os.environ["SWANLAB_PROJECT"] = "qwen3-sft-financial"
PROMPT = "You are a financial expert. Analyze the given financial data or question and provide a well-reasoned response."
MAX_LENGTH = 512

swanlab.config.update({
    "model": "Qwen/Qwen3-0.6B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
})


def process_func(example):
    """
    Preprocess dataset examples into tokenized input/output pairs.
    The finance-alpaca data already has instruction/input/output fields.
    We use PROMPT as system message, combine instruction + input as user message.
    """
    input_ids, attention_mask, labels = [], [], []

    # Build user content: instruction is the question, input is optional context
    user_content = example["instruction"]
    if example.get("input", ""):
        user_content += f"\n{example['input']}"

    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = next(model.parameters()).device
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# Download Qwen model from modelscope
model_dir = snapshot_download("Qwen/Qwen3-0.6B", cache_dir="./", revision="master")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
model.enable_input_require_grads()

# LoRA config
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, config)

# Load datasets directly — finance-alpaca data already has instruction/input/output
train_dataset_path = "train.jsonl"
test_dataset_path = "val.jsonl"

train_df = pd.read_json(train_dataset_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

eval_df = pd.read_json(test_dataset_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

args = TrainingArguments(
    output_dir="./output/Qwen3-0.6B-financial",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=400,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="swanlab",
    run_name="qwen3-0.6B-financial",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# Quick evaluation: test with first 3 examples from val set
test_df = pd.read_json(test_dataset_path, lines=True)[:3]

test_text_list = []

for index, row in test_df.iterrows():
    user_content = row["instruction"]
    if row.get("input", ""):
        user_content += f"\n{row['input']}"

    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": user_content}
    ]

    response = predict(messages, model, tokenizer)

    response_text = f"""
    Question: {user_content}

    LLM: {response}
    """

    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

swanlab.log({"Prediction": test_text_list})

swanlab.finish()