import re
from flask import Flask, render_template, request
import numpy as np
from scipy.sparse import load_npz
from datasets import load_dataset

app = Flask(__name__)

class ArxivSearch:
    def __init__(self, dataset, embedding="tfidf"):
        self.documents = []
        self.titles = []
        self.raw_texts = []
        self.arxiv_ids = []

        self.load_data(dataset)
        if embedding == "tfidf":
            self.tfidf_matrix = load_npz("TF-IDF embeddings/tfidf_matrix_small.npz")
            with open("TF-IDF embeddings/feature_names_small.txt", "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]

    def load_data(self, dataset):
        train_data = dataset["train"]

        for item in train_data.select(range(len(train_data))):
            text = item["text"]
            if not text or len(text.strip()) < 10:
                continue

            lines = text.splitlines()
            title_lines = []
            found_arxiv = False
            arxiv_id = None

            for line in lines:
                line_strip = line.strip()
                if not found_arxiv and line_strip.lower().startswith("arxiv:"):
                    found_arxiv = True
                    match = re.search(r'arxiv:\d{4}\.\d{4,5}v\d', line_strip, flags=re.IGNORECASE)
                    if match:
                        arxiv_id = match.group(0).lower()
                elif not found_arxiv:
                    title_lines.append(line_strip)
                else:
                    if line_strip.lower().startswith("abstract"):
                        break

            title = " ".join(title_lines).strip()
            self.raw_texts.append(text.strip())
            self.titles.append(title)
            self.documents.append(text.strip())
            self.arxiv_ids.append(arxiv_id)

    def keyword_match_ranking(self, query, top_n=5):
        query_terms = query.lower().split()
        query_indices = [i for i, term in enumerate(self.feature_names) if term in query_terms]
        if not query_indices:
            return []

        scores = []
        for doc_idx in range(self.tfidf_matrix.shape[0]):
            doc_vector = self.tfidf_matrix[doc_idx]
            doc_score = sum(doc_vector[0, i] for i in query_indices)
            if doc_score > 0:
                scores.append((doc_idx, doc_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    @staticmethod
    def snippet_before_abstract(text):
        pattern = re.compile(r'a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|i\s*n\s*t\s*r\s*o\s*d\s*u\s*c\s*t\s*i\s*o\s*n', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return text[:match.start()].strip()
        else:
            return text[:100].strip()


# Load dataset and initialize search engine
dataset = load_dataset("ccdv/arxiv-classification", "no_ref")  # replace with your dataset
search_engine = ArxivSearch(dataset)

@app.route("/", methods=["GET", "POST"])
def index():
    query = request.form.get("query", "")
    results = []
    if query:
        ranked = search_engine.keyword_match_ranking(query)
        for idx, score in ranked:
            if not search_engine.arxiv_ids[idx]:
                continue
            arxiv_id = search_engine.arxiv_ids[idx].replace("arxiv:", "")
            link = f"https://arxiv.org/abs/{arxiv_id}"
            snippet = search_engine.snippet_before_abstract(search_engine.raw_texts[idx])
            results.append({
                "link": link,
                "snippet": snippet
            })

    return render_template("index.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)
