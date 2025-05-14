from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

# Load the dataset
ds = load_dataset("ccdv/arxiv-classification", "no_ref")

# Dictionary mapping from numeric labels to class names
arxiv_subjects = {
    '0': 'Commutative Algebra',
    '1': 'Computer Vision',
    '2': 'Artificial Intelligence',
    '3': 'Systems and Control',
    '4': 'Group Theory',
    '5': 'Computational Engineering',
    '6': 'Programming Languages',
    '7': 'Information Theory',
    '8': 'Data Structures',
    '9': 'Neural and Evolutionary',
    '10': 'Statistics Theory'
}

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define batch size
batch_size = 64

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)

# Prepare the dataset
documents = ds["train"][:]['text']

# Initialize a list to store embeddings
cls_embeddings = []

# Process documents in batches with a progress bar
for i in tqdm(range(0, len(documents), batch_size), desc="Processing Batches"):
    batch = documents[i:i + batch_size]
    
    # Tokenize the input text
    inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length = 512).to(device)
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract the embedding for the [CLS] token (first token)
        cls_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())  # Move to CPU to save memory

# Concatenate all embeddings
cls_embedding = torch.cat(cls_embeddings, dim=0)

# save bert embeddings
np.savez_compressed("scibert_embedding.npz", 
                   bert_embedding=cls_embedding.numpy())



# Prepare the dataset
documents = ds["test"][:]['text']

pre_truncated_docs = [doc[doc.index('Abstract'):] if 'Abstract' in doc else doc for doc in documents]

# Initialize a list to store embeddings
cls_embedding_test = []

# Process documents in batches with a progress bar
for i in tqdm(range(0, len(documents), batch_size), desc="Processing Batches"):
    batch = documents[i:i + batch_size]
    
    # Tokenize the input text
    inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length = 512).to(device)
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract the embedding for the [CLS] token (first token)
        cls_embedding_test.append(outputs.last_hidden_state[:, 0, :].cpu())  # Move to CPU to save memory

# Concatenate all embeddings
cls_embedding_test = torch.cat(cls_embedding_test, dim=0)

# save bert embeddings
np.savez_compressed("scibert_embedding_test.npz", 
                   bert_embedding=cls_embedding_test.numpy())
