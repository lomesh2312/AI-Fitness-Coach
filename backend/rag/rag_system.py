import json
import os
import gc
import threading
import numpy as np

# ── Lazy import helpers ──────────────────────────────────────────────────────
_faiss_module   = None
_st_model       = None
_model_lock     = threading.Lock()

def _get_faiss():
    global _faiss_module
    if _faiss_module is None:
        import faiss
        _faiss_module = faiss
    return _faiss_module

def _get_model():
    global _st_model
    if _st_model is not None:
        return _st_model
    with _model_lock:
        if _st_model is None:
            from sentence_transformers import SentenceTransformer
            # all-MiniLM-L6-v2 is 22MB — small but high quality
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ [RAG] SentenceTransformer model loaded.")
            gc.collect()
    return _st_model


# ── Keyword-based fallback retrieval (no FAISS needed) ──────────────────────
def _keyword_search(query: str, documents: list, top_k: int = 3) -> list:
    """Fallback: return documents containing any word from the query."""
    q_words = set(query.lower().split())
    scored  = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        score = len(q_words & doc_words)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


class RAGSystem:
    def __init__(self, knowledge_path: str):
        self.knowledge_path = knowledge_path
        self.documents: list = []
        self.index          = None
        self._built         = False
        self._build_lock    = threading.Lock()  # Prevent race conditions

    def _build_index(self):
        """Build FAISS index on first use. Thread-safe via lock."""
        if self._built:
            return
        with self._build_lock:
            if self._built:   # Double-checked locking
                return
            if not os.path.exists(self.knowledge_path):
                print(f"⚠️  [RAG] Knowledge base not found at {self.knowledge_path}")
                self._built = True
                return

            with open(self.knowledge_path, "r") as f:
                data = json.load(f)

            # Only load the first 4 categories (fitness/diet/exercise/food_nutrition subset)
            # to stay within memory limits on the free tier
            priority_keys = ["fitness_rules", "diet_knowledge", "exercise_science", "food_nutrition"]
            for key in priority_keys:
                if key in data:
                    entries = data[key]
                    # Cap food_nutrition at 200 entries to save memory
                    if key == "food_nutrition":
                        entries = entries[:200]
                    self.documents.extend(entries)

            print(f"📝 [RAG] Loaded {len(self.documents)} documents. Building FAISS index...")

            try:
                model = _get_model()
                embeddings = model.encode(
                    self.documents,
                    batch_size=32,
                    show_progress_bar=False
                )
                faiss_mod = _get_faiss()
                dimension = embeddings.shape[1]
                index     = faiss_mod.IndexFlatL2(dimension)
                index.add(np.array(embeddings, dtype="float32"))
                self.index = index
                print(f"✅ [RAG] FAISS index built with {len(self.documents)} entries.")
            except Exception as e:
                print(f"❌ [RAG] FAISS build failed: {e}. Will use keyword fallback.")
                self.index = None

            self._built = True
            gc.collect()

    def retrieve(self, query: str, top_k: int = 3) -> list:
        self._build_index()

        # ── FAISS path ───────────────────────────────────────────────────────
        if self.index is not None:
            try:
                model = _get_model()
                q_emb  = model.encode([query])
                dists, idxs = self.index.search(
                    np.array(q_emb, dtype="float32"), top_k
                )
                results = [self.documents[i] for i in idxs[0] if i < len(self.documents)]
                print(f"📚 [RAG] FAISS retrieved {len(results)} results.")
                return results
            except Exception as e:
                print(f"⚠️  [RAG] FAISS search error: {e}. Using keyword fallback.")

        # ── Keyword fallback ─────────────────────────────────────────────────
        if self.documents:
            results = _keyword_search(query, self.documents, top_k)
            print(f"📚 [RAG] Keyword fallback retrieved {len(results)} results.")
            return results

        return []


BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
rag_instance = RAGSystem(os.path.join(BASE_DIR, "knowledge_base.json"))
