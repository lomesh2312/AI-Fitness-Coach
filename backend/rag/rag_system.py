import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

class RAGSystem:
    def __init__(self, knowledge_path):
        self.knowledge_path = knowledge_path
        self.documents = []
        self.index = None
        self.load_and_index()

    def load_and_index(self):
        if not os.path.exists(self.knowledge_path):
            print(f"Knowledge base not found at {self.knowledge_path}")
            return
            
        with open(self.knowledge_path, 'r') as f:
            data = json.load(f)
            
        for category in data:
            self.documents.extend(data[category])
            
        embeddings = model.encode(self.documents)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"RAG System: Indexed {len(self.documents)} facts.")

    def retrieve(self, query, top_k=2):
        if self.index is None:
            return []
            
        query_embedding = model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = [self.documents[i] for i in indices[0]]
        return results

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rag_instance = RAGSystem(os.path.join(BASE_DIR, "knowledge_base.json"))
