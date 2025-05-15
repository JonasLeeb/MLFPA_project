import torch
import numpy as np

def embed_chunk(batch, tokenizer, model, device, max_length=512, max_chunks=None):
    embeddings = []
    
    # Tokenize the batch with attention mask
    tokens = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
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