from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

# Load the dataset
ds = load_dataset("ccdv/arxiv-classification", "no_ref")

ds = ds['train'][:]['text'].replace('\n', ' ').strip()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define batch size
batch_size = 16

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Prepare the dataset
documents = ds["train"][:]['text']

# Initialize a list to store embeddings
cls_embeddings = []

# Process documents in batches with a progress bar
for i in tqdm(range(0, len(documents), batch_size), desc="Processing Batches"):
    batch = documents[i:i + batch_size]
    
    # Tokenize the input text
    inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract the embedding for the [CLS] token (first token)
        cls_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())  # Move to CPU to save memory

# Concatenate all embeddings
cls_embedding = torch.cat(cls_embeddings, dim=0)

# save bert embeddings
np.savez_compressed("pre_bert_embedding.npz", 
                   bert_embedding=cls_embedding.numpy())