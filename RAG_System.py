from dataclasses import dataclass
from typing import List
import chromadb
import ollama

@dataclass
class RAGConfig:
    collection_name: str = "company_knowledge_base"
    model_name: str = "llama3.1:8b"  
    top_k: int = 2

class LocalKnowledgeEngine:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name=config.collection_name)

    def ingest_documents(self, documents: List[str]) -> None:
        ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(documents=documents, ids=ids)

    def query(self, user_query: str) -> str:
        results = self.collection.query(
            query_texts=[user_query],
            n_results=self.config.top_k
        )
        retrieved_docs = results['documents'][0] if results['documents'] else []
        context = "\n---\n".join(retrieved_docs)

        system_prompt = (
            "You are an enterprise helpdesk assistant. "
            "Answer strictly using the provided context. If the answer cannot be found, "
            "state: 'Information not available in internal records.'\n\n"
            f"Context:\n{context}"
        )

        response = ollama.chat(
            model=self.config.model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_query}
            ]
        )
        return response['message']['content']