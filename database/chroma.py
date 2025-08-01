import chromadb
from typing import List, Dict, Any
from .databaseInterface import DatabaseInterface

class ChromaDB(DatabaseInterface):
    def __init__(self, collection_name: str = "youtube_transcripts", persist_path: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_path)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name, 
                metadata={
                    "description": "YouTube Transcripts",
                    "hnsw:space": "cosine",
                }
            )
    
    def add_embedding(self, id: str, embedding: List[float], document: str, metadata: Dict[str, Any]):
        self.collection.add(
            embeddings=[embedding],
            ids=[id],
            documents=[document],
            metadatas=[metadata]
        )
    
    def find_similar_embeddings(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
    
    def get_document(self, id: str) -> str:
        result = self.collection.get(ids=[id])
        return result['documents'][0] if result['documents'] else None
    
    def get_metadata(self, id: str) -> Dict[str, Any]:
        result = self.collection.get(ids=[id])
        return result['metadatas'][0] if result['metadatas'] else None