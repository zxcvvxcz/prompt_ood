from transformers import GPT2Tokenizer, GPT2Model
import torch
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2Model.from_pretrained('gpt2-large')
txt = "Hello, my dog is cute"
inputs = tokenizer(txt, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(txt)
print(last_hidden_states.shape)
