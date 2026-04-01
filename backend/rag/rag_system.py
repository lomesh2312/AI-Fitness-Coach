import json
import os
import faiss
import numpy as np
import gc

MODEL_NAME = 'all-MiniLM-L6-v2'
_model = None  # LAZY LOAD TO FIX DEPLOYMENT MEMORY CRASH

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        gc.collect()
    return _model

class RAGSystem:
    def __init__(self, knowledge_path):
        self.knowledge_path = knowledge_path
        self.documents = []
        self.index = None
        # DELAY LOAD AND INDEX UNTIL FIRST REQUEST TO SAVE RAM ON STARTUP

    def load_and_index(self):
        if self.index is not None:
            return # Already loaded
            
        if not os.path.exists(self.knowledge_path):
            print(f"Knowledge base not found at {self.knowledge_path}")
            return
            
        with open(self.knowledge_path, 'r') as f:
            data = json.load(f)
            
        for category in data:
            self.documents.extend(data[category])
            
        model = get_model()
        embeddings = model.encode(self.documents)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"RAG System: Indexed {len(self.documents)} facts.")
        gc.collect()

    def retrieve(self, query, top_k=2):
        if self.index is None:
            self.load_and_index() # LAZY LOAD HERE
            
        if self.index is None:
            return []
            
        model = get_model()
        query_embedding = model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = [self.documents[i] for i in indices[0]]
        return results

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rag_instance = RAGSystem(os.path.join(BASE_DIR, "knowledge_base.json"))
