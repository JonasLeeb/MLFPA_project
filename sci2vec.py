import os
import urllib.request
from gensim.models import KeyedVectors
from scipy.sparse import load_npz
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


def download_biowordvec(destination_path):
    url = "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin"
    print("Downloading BioWordVec model...")
    urllib.request.urlretrieve(url, destination_path)
    print("Download completed.")

def load_biowordvec_model(model_path):
    print("Loading BioWordVec model...")
    #using a word limit because not enough dedicated wam 
    wv = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=None)
    print("Model loaded successfully.")
    return wv

if __name__ == "__main__":
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "BioWordVec_PubMed_MIMICIII_d200.vec.bin")

    if not os.path.isfile(model_path):
        download_biowordvec(model_path)

    wv = load_biowordvec_model(model_path)

    # Load the dataset
    ds = load_dataset("ccdv/arxiv-classification", "no_ref")

    # # Load TF-IDF matrix and feature names if not already loaded
    # try:
    #     tfidf_matrix_train
    # except NameError:
    #     print("TF-IDF matrix is not loaded. Loading now...")
    #     tfidf_matrix_train = load_npz("TF-IDF embeddings/tfidf_matrix_train.npz")
    #     with open("TF-IDF embeddings/feature_names.txt", "r") as f:
    #         feature_names = [line.strip() for line in f.readlines()]

    # good_words = [word for word in feature_names if word in wv.key_to_index]
    # bad_words = [word for word in feature_names if word not in wv.key_to_index]

    # print(f"Number of good words: {len(good_words)}")
    # print(f"Number of bad words: {len(feature_names) - len(good_words)}")
    # print(f"Number of words in the model: {len(wv.key_to_index)}")

    # print('bad words:', bad_words[:10])

    # # Initialize Word2Vec embedding matrix
    # word2vec_embedding = np.zeros((tfidf_matrix_train.shape[0], 200))

    # # Load dataset
    # print("Loading dataset...")
    # texts = ds["train"][:]['text']
    # print("Dataset loaded successfully!")

    # # Precompute TF-IDF dictionaries for each document
    # print("Preprocessing TF-IDF weights...")
    # from scipy.sparse import coo_matrix

    # tfidf_dicts = []
    # for i in tqdm(range(tfidf_matrix_train.shape[0]), desc="Creating TF-IDF dictionaries"):
    #     row = tfidf_matrix_train[i].tocoo()
    #     words_in_doc = [feature_names[idx] for idx in row.col]
    #     tfidf_dicts.append(dict(zip(words_in_doc, row.data)))

    # print("Calculating weighted embeddings...")
    # for i, text in enumerate(tqdm(texts, desc="Processing documents")):
    #     words = [word for word in text.split() if word in wv.key_to_index]
    #     if not words:
    #         continue  # Keep as zeros if no valid words

    #     # Get vectors and weights
    #     vectors = np.stack([wv[word] for word in words])
    #     weights = np.array([tfidf_dicts[i].get(word, 0.0) for word in words])

    #     # Calculate weighted average with numerical stability
    #     try:
    #         weighted_avg = np.average(vectors, axis=0, weights=weights)
    #     except ZeroDivisionError:
    #         weighted_avg = np.mean(vectors, axis=0)

    #     word2vec_embedding[i] = weighted_avg

    # # Save results
    # print("Saving embeddings...")
    # np.savez_compressed("sci2vec_embedding.npz", 
    #                 word2vec_embedding=word2vec_embedding)
    # print("Embeddings saved successfully!")

    # Load TF-IDF matrix and feature names if not already loaded
    try:
        tfidf_matrix_test
    except NameError:
        print("TF-IDF matrix is not loaded. Loading now...")
        tfidf_matrix_test = load_npz("TF-IDF embeddings/tfidf_matrix_train.npz")
        with open("TF-IDF embeddings/feature_names.txt", "r") as f:
            feature_names = [line.strip() for line in f.readlines()]

    # Initialize Word2Vec embedding matrix
    word2vec_embedding_test = np.zeros((tfidf_matrix_test.shape[0], 200))

    # Load dataset
    print("Loading dataset...")
    texts = ds["test"][:]['text']
    print("Dataset loaded successfully!")

    # Preprocess texts
    print("Preprocessing texts...")
    tokenized_texts = [text.split() for text in texts]
    filtered_texts = [[word for word in text if word in wv.key_to_index] 
                    for text in tokenized_texts]
    print("Text preprocessing completed!")

    # Precompute feature index map
    feature_index = {word: idx for idx, word in enumerate(feature_names)}

    # Precompute word vector cache
    print("Caching word vectors...")
    word_vector_cache = {word: wv[word] for word in wv.key_to_index}

    # Precompute TF-IDF dictionaries for each document
    print("Preprocessing TF-IDF weights...")
    from scipy.sparse import coo_matrix

    tfidf_dicts = []
    for i in tqdm(range(tfidf_matrix_test.shape[0]), desc="Creating TF-IDF dictionaries"):
        row = tfidf_matrix_test[i].tocoo()
        words_in_doc = [feature_names[idx] for idx in row.col]
        tfidf_dicts.append(dict(zip(words_in_doc, row.data)))

    print("Calculating weighted embeddings...")
    for i, text in enumerate(tqdm(texts, desc="Processing documents")):
        words = [word for word in text.split() if word in wv.key_to_index]
        if not words:
            continue  # Keep as zeros if no valid words

        # Get vectors and weights
        vectors = np.stack([wv[word] for word in words])
        weights = np.array([tfidf_dicts[i].get(word, 0.0) for word in words])

        # Calculate weighted average with numerical stability
        try:
            weighted_avg = np.average(vectors, axis=0, weights=weights)
        except ZeroDivisionError:
            weighted_avg = np.mean(vectors, axis=0)

        word2vec_embedding_test[i] = weighted_avg

    # Save results
    print("Saving embeddings...")
    np.savez_compressed("sci2vec_embedding_test.npz", 
                    word2vec_embedding_test=word2vec_embedding_test)
    print("Embeddings saved successfully!")