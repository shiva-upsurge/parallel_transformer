from src.models.modeling_parallel_gpt2 import ParallelGPT2LMHeadModel, ParallelGPT2Config
from transformers import AutoModel, AutoConfig
import torch
# config = AutoConfig.from_pretrained("output/models/drift-gpt2-medium-wikitext", trust_remote_code=True)
model = AutoModel.from_pretrained("output/models/parallel-gpt2-medium-wikitext", trust_remote_code=True)

model(torch.randint(0, 10000, (1, 100)))
print()


