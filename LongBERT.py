from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from helper_functions import embed_chunk

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
batch_size = 16
num_chunks = 10

# Initializing a model from the modernbert-base style configuration
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Prepare the dataset
documents = ds["train"][:]['text']

# Initialize a list of lists to collect embeddings for each chunk
cls_embeddings = [[] for _ in range(num_chunks)]

# Process documents in batches
for i in tqdm(range(0, len(documents), batch_size), desc="Processing Batches"):

    batch = documents[i:i + batch_size]
    embedded_chunk = embed_chunk(batch, max_length=512, max_chunks=num_chunks, tokenizer=tokenizer, model=model, device=device)
    for j, chunk in enumerate(embedded_chunk):
        cls_embeddings[j].append(chunk)

# Concatenate along the batch dimension for each chunk
cls_embeddings = [torch.cat(chunks, axis=0) for chunks in cls_embeddings if chunks]
# save bert embeddings
np.savez_compressed("longbert_embedding.npz", 
                   bert_embedding=cls_embeddings.numpy())



# Prepare the dataset
documents = ds["test"][:]['text']

pre_truncated_docs = [doc[doc.index('Abstract'):] if 'Abstract' in doc else doc for doc in documents]

# Initialize a list of lists to collect embeddings for each chunk
cls_embeddings_test = [[] for _ in range(num_chunks)]

# Process documents in batches
for i in tqdm(range(0, len(documents), batch_size), desc="Processing Batches"):
    batch = documents[i:i + batch_size]
    embedded_chunk = embed_chunk(batch, max_length=512, max_chunks=num_chunks, tokenizer=tokenizer, model=model, device=device)
    for j, chunk in enumerate(embedded_chunk):
        cls_embeddings_test[j].append(chunk)

# Concatenate along the batch dimension for each chunk
cls_embeddings_test = [torch.cat(chunks, axis=0) for chunks in cls_embeddings if chunks]

# save bert embeddings
np.savez_compressed("longbert_embedding_test.npz", 
                   bert_embedding=cls_embeddings_test.numpy())