from abc import ABC, abstractmethod
from typing import List, Dict, Any

class DatabaseInterface(ABC):
    @abstractmethod
    def add_embedding(self, id: str, embedding: List[float], document: str, metadata: Dict[str, Any]):
        pass
    
    @abstractmethod
    def find_similar_embeddings(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_document(self, id: str) -> str:
        pass
    
    @abstractmethod
    def get_metadata(self, id: str) -> Dict[str, Any]:
        pass