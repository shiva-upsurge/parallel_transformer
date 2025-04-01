from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import random

# Initialize model and tokenizer
model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

# Example text
text = "The cat sits on the mat."
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Create labels by copying input_ids
labels = input_ids.clone()

# Randomly mask 15% of the tokens
mask_probability = 0.15
mask_token_id = tokenizer.mask_token_id

# Select random positions to mask (excluding special tokens like [CLS], [SEP])
special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id]
candidate_mask_positions = [
    i for i in range(len(input_ids[0]))
    if input_ids[0][i] not in special_tokens
]

num_tokens_to_mask = max(1, int(len(candidate_mask_positions) * mask_probability))
mask_positions = random.sample(candidate_mask_positions, num_tokens_to_mask)

# Apply masking
for pos in mask_positions:
    input_ids[0][pos] = mask_token_id
    # labels already contains the original tokens

# Forward pass with masked input and labels
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

# Get loss and predictions
loss = outputs.loss
logits = outputs.logits

# Get predictions for masked positions
predictions = torch.argmax(logits, dim=-1)

print(f"Original text: {text}")
print(f"Masked text: {tokenizer.decode(input_ids[0])}")
print(f"Loss: {loss.item()}")

# Print predictions for masked tokens
for pos in mask_positions:
    predicted_token = tokenizer.decode([predictions[0][pos]])
    original_token = tokenizer.decode([labels[0][pos]])
    print(f"Position {pos}:")
    print(f"  Original: {original_token}")
    print(f"  Predicted: {predicted_token}")