import json
import os
import gc
import threading
import numpy as np

# --- STEP 1: LAZY LOADING ---
# We keep these as None at startup.
# This prevents the 600MB+ memory spike when the server first boots.
_faiss_module = None
_st_model = None
_model_lock = threading.Lock()


def _get_faiss():
    global _faiss_module
    if _faiss_module is None:
        import faiss  # Import only when needed

        _faiss_module = faiss
    return _faiss_module


def _get_model():
    global _st_model
    if _st_model is not None:
        return _st_model
    with _model_lock:
        if _st_model is None:
            from sentence_transformers import SentenceTransformer

            # all-MiniLM-L6-v2 is small (22MB disk) but takes ~100MB RAM to run
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("🧠 [RAG] SentenceTransformer model loaded into memory.")
            gc.collect()  # Force free unused memory
    return _st_model


# --- STEP 2: SAFE KEYWORD FALLBACK ---
def _keyword_search(query: str, documents: list, top_k: int = 3) -> list:
    """
    If FAISS ever crashes due to memory limits, we fall back to a simple,
    0-memory keyword match so the system NEVER crashes.
    """
    q_words = set(query.lower().split())
    scored = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        score = len(q_words & doc_words)
        if score > 0:
            scored.append((score, doc))
    # Sort by highest matching words
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


class RAGSystem:
    def __init__(self, knowledge_path: str):
        self.knowledge_path = knowledge_path
        self.documents = []
        self.index = None
        self._built = False
        self._build_lock = (
            threading.Lock()
        )  # Prevents 2 users from building index at same time

    def _build_index_safe(self):
        """Builds the FAISS memory index safely, capping at 200 items."""
        if self._built:
            return

        with self._build_lock:
            # Double-check inside lock
            if self._built:
                return

            if not os.path.exists(self.knowledge_path):
                print(f"⚠️ [RAG] Knowledge base not found at {self.knowledge_path}")
                self._built = True
                return

            with open(self.knowledge_path, "r") as f:
                data = json.load(f)

            # --- STEP 3: MEMORY CAPPING ---
            # Instead of loading 1000+ entries (which causes a 600MB spike),
            # we safely cap combined data to roughly 200 core facts.
            priority_categories = [
                "fitness_rules",
                "diet_knowledge",
                "exercise_science",
                "food_nutrition",
            ]

            for key in priority_categories:
                if key in data:
                    entries = data[key]
                    # The food_nutrition array alone is massively huge. Cap it.
                    if key == "food_nutrition":
                        entries = entries[:150]  # 150 items max
                    else:
                        entries = entries[:25]  # 25 items max for others

                    self.documents.extend(entries)

            print(
                f"📚 [RAG] Capped knowledge base to {len(self.documents)} total entries to save RAM."
            )

            try:
                # Encode ONLY the 200 entries (Takes ~2 seconds, uses minimal RAM)
                model = _get_model()
                embeddings = model.encode(self.documents, show_progress_bar=False)

                faiss_mod = _get_faiss()
                dimension = embeddings.shape[1]
                self.index = faiss_mod.IndexFlatL2(dimension)
                self.index.add(np.array(embeddings, dtype="float32"))
                print(f"✅ [RAG] FAISS Index built successfully.")
            except Exception as e:
                print(
                    f"❌ [RAG] FAISS build failed ({e}). System will survive using Keyword Fallback."
                )
                self.index = None

            self._built = True
            gc.collect()  # Clean up the encoding mess instantly

    def retrieve(self, query: str, top_k: int = 2) -> list:
        # 1. Trigger the safe build (only happens once on the very first API hit)
        self._build_index_safe()

        # 2. Try the Smart FAISS search first
        if self.index is not None:
            try:
                model = _get_model()
                query_embedding = model.encode([query])
                distances, indices = self.index.search(
                    np.array(query_embedding, dtype="float32"), top_k
                )

                results = [
                    self.documents[i] for i in indices[0] if i < len(self.documents)
                ]
                return results
            except Exception as e:
                print(
                    f"⚠️ [RAG] FAISS search error: {e}. Switching to safe keyword mode."
                )

        # 3. If FAISS failed/OOM, use the zero-memory keyword search
        if self.documents:
            return _keyword_search(query, self.documents, top_k)

        # 4. Ultimate absolute fallback (should never happen)
        return ["Eating protein and staying hydrated is essential for fitness."]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rag_instance = RAGSystem(os.path.join(BASE_DIR, "knowledge_base.json"))
