import torch
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import nltk
nltk.download('all')
from nltk.tokenize import sent_tokenize

def embed_chunk(batch, tokenizer, model, device, max_length=512, max_chunks=None):
    embeddings = []
    
    # Tokenize the batch with attention mask
    tokens = tokenizer(
        batch,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length*max_chunks,
        truncation=True
    )
    
    # Extract input IDs and attention mask
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    # Determine the number of chunks
    num_chunks = max_chunks if max_chunks else len(input_ids[0]) // max_length * 2
    chunks = [
        (input_ids[:, i:i + max_length], attention_mask[:, i:i + max_length])
        for i in range(0, num_chunks * int(max_length / 2), int(max_length / 2))
    ]
    
    # Process each chunk
    for chunk_input_ids, chunk_attention_mask in chunks:
        chunk_input_ids = chunk_input_ids.to(device)
        chunk_attention_mask = chunk_attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)
            # Use the mean of the last hidden state as the embedding
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu())
    
    return embeddings



def describe_structure(obj, depth=0, max_items=3):
    indent = "  " * depth
    prefix = f"{indent}- "

    if isinstance(obj, torch.Tensor):
        print(f"{prefix}torch.Tensor shape={tuple(obj.shape)}, dtype={obj.dtype}")
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}np.ndarray shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, list):
        print(f"{prefix}list len={len(obj)}")
        for i, item in enumerate(obj[:max_items]):
            print(f"{indent}  [{i}]:")
            describe_structure(item, depth + 2, max_items)
        if len(obj) > max_items:
            print(f"{indent}  [...{len(obj) - max_items} more items]")
    elif isinstance(obj, dict):
        print(f"{prefix}dict len={len(obj)}")
        for i, (k, v) in enumerate(list(obj.items())[:max_items]):
            print(f"{indent}  [{repr(k)}]:")
            describe_structure(v, depth + 2, max_items)
        if len(obj) > max_items:
            print(f"{indent}  [...{len(obj) - max_items} more items]")
    else:
        print(f"{prefix}{type(obj).__name__}: {repr(obj)}")

def PCA_encodings(embedding, title, ds):
    # Assuming `data` is your 300-dimensional dataset (shape: [n_samples, 300])
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(embedding[:10000])


    colors = np.array(ds["train"][:]['label'])[:10000]
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        z=reduced_data[:, 2],
        mode='markers+text',
        marker=dict(
            size=3.5,
            color=colors,
            colorscale='turbo',
            opacity=0.5
        ),
        # text=titles[:200],
    )])
    fig.update_layout(
        title=title,
    )

    fig.update_layout(
        # title=f"3D Scatter Plot of Word Vectors for '{center_word}'",
        scene=dict(
            xaxis_title='PCA 1',
            yaxis_title='PCA 2',
            zaxis_title='PCA 3'
        ),
        coloraxis_colorbar=dict(
            title="Similarity",
            thickness=20,
            len=0.75,
            x=1.1  # Position the colorbar slightly outside the plot
        ),
        width=1000,
        height=800,
    )
    fig.show(renderer="browser")


def encode_long_document(text, model, chunk_size=5):
    device = model.device
    sentences = sent_tokenize(text)
    chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, min(len(sentences),50), chunk_size)]
    # If using SentenceTransformers:
    chunk_embeddings = model.encode(chunks, device=device)
    doc_embedding = np.mean(chunk_embeddings, axis=0)
    return doc_embedding