#!/usr/bin/env python3

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

base = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base)

model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "./amstrad-model")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

messages = [
    {
        "role": "user",
        "content": "Which CPU is used in the Amstrad PCW?"
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

result = pipe(
    prompt,
    max_new_tokens=50,
    do_sample=False,
)

print(result[0]["generated_text"])
