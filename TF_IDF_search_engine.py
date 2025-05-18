import re
from sklearn.feature_extraction.text import TfidfVectorizer
from ipywidgets import widgets, HBox, Output
import IPython.display as display
from scipy.sparse import load_npz
import numpy as np



class ArxivSearch:
    def __init__(self, dataset, encoding):
        """
        Initializes the search engine with the provided dataset and encoding.

        Args:
            dataset (str): Path to the dataset file to load.
            encoding (str): The encoding method to use for document representation.
                Possible values:
                    - "tfidf": Use precomputed TF-IDF embeddings.

        Attributes:
            documents (list): List of processed document texts.
            titles (list): List of document titles.
            raw_texts (list): List of raw document texts.
            arxiv_ids (list): List of arXiv IDs for the documents.
            tfidf_matrix (scipy.sparse matrix, optional): TF-IDF matrix if encoding is "tfidf".
            feature_names (list, optional): List of feature names for the TF-IDF matrix.
            search_box (widgets.Text): Text input widget for search queries.
            search_button (widgets.Button): Button widget to trigger search.
            output_area (widgets.Output): Output widget to display search results.

        Note:
            Only "tfidf" encoding is currently supported.
        """
        
        self.documents = []
        self.titles = []
        self.raw_texts = []
        self.arxiv_ids = []

        self.load_data(dataset)
        # self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        # self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        if encoding == "tfidf":
            self.tfidf_matrix = load_npz("TF-IDF embeddings/tfidf_matrix_train.npz")
            with open("TF-IDF embeddings/feature_names.txt", "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]

        # Widgets
        self.search_box = widgets.Text(
            value='',
            placeholder='Type your search query here',
            description='Search:',
            layout=widgets.Layout(width='70%')
        )
        self.search_button = widgets.Button(
            description='Search',
            button_style='primary',
            layout=widgets.Layout(width='20%')
        )
        self.output_area = Output(layout={'border': '1px solid #ccc', 'padding': '10px', 'margin-top': '10px'})

        self.search_button.on_click(self.display_results_on_click)

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
                    # Skip authors and stop at abstract (you can extend this if needed)
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

    def display_results_on_click(self, b):
        print(len(self.arxiv_ids))
        query = self.search_box.value.strip()
        self.output_area.clear_output()
        if not query:
            return
        results = self.keyword_match_ranking(query)
        with self.output_area:
            if not results:
                display.display(widgets.HTML("<p>No results found.</p>"))
                return
            display_rank = 1
            for idx, score in results:
                if not self.arxiv_ids[idx]:
                    continue  # Skip documents without arXiv links

                display.display(widgets.HTML(f"<h3>Document {display_rank}</h3>"))
                display_rank += 1

                arxiv_num = self.arxiv_ids[idx].replace('arxiv:', '')
                link = f"https://arxiv.org/abs/{arxiv_num}"
                display.display(widgets.HTML(f'<b>arXiv Link:</b> <a href="{link}" target="_blank">{link}</a>'))

                snippet = self.snippet_before_abstract(self.raw_texts[idx]).replace('\n', '<br>')
                display.display(widgets.HTML(f"<pre>{snippet}</pre>"))
                display.display(widgets.HTML("<hr>"))

    def show_widgets(self):
        display.display(HBox([self.search_box, self.search_button]), self.output_area)