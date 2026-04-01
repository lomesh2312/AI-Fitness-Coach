import json
import os
import gc
import numpy as np

# ─── Lazy imports to defer heavy memory usage until first request ───────────
_faiss = None
_st_model = None

def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss

def _get_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        # Use a tiny model to minimise RAM footprint (~22MB vs 90MB default)
        _st_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        gc.collect()
    return _st_model


class RAGSystem:
    def __init__(self, knowledge_path):
        self.knowledge_path = knowledge_path
        self.documents = []
        self.index = None
        self._built = False

    def _build_index(self):
        """Build FAISS index on first use, not on import."""
        if self._built:
            return
        if not os.path.exists(self.knowledge_path):
            print(f"Knowledge base not found at {self.knowledge_path}")
            self._built = True
            return

        with open(self.knowledge_path, 'r') as f:
            data = json.load(f)

        for category in data:
            self.documents.extend(data[category])

        model = _get_model()
        embeddings = model.encode(self.documents, batch_size=16, show_progress_bar=False)

        faiss = _get_faiss()
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        self.index = index
        print(f"RAG System: Indexed {len(self.documents)} facts.")
        self._built = True
        gc.collect()

    def retrieve(self, query, top_k=2):
        self._build_index()  # Build lazily on first request
        if self.index is None:
            return []

        model = _get_model()
        query_embedding = model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        results = [self.documents[i] for i in indices[0]]
        return results


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rag_instance = RAGSystem(os.path.join(BASE_DIR, "knowledge_base.json"))
