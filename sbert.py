from sentence_transformers import SentenceTransformer
from helper_functions import encode_long_document
from tqdm import tqdm
from datasets import load_dataset
import numpy as np


dataset = load_dataset("ccdv/arxiv-classification", "no_ref")


model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda')
sent_encoding = []

for doc in tqdm(dataset["train"][:]['text'], desc="Encoding documents"):
    sent_encoding.append(encode_long_document(doc, model, chunk_size=5))
# 1. Load a pretrained Sentence Transformer model

# save the embeddings
np.savez_compressed("sbert_embedding.npz", 
                   sbert_embedding=sent_encoding)