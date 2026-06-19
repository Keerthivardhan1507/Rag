from src.data_loader import read_all_documents
from src.vector_store import vectorstore
from src.search import RAGsearch
import os


if __name__ =="__main__":
    docs = read_all_documents("data")
    store = vectorstore("faiss_store")
    rag_search = RAGsearch()
    query = "what is EDA"
    summary = rag_search.search_and_summarize(query,top_k=3)
    print("summary",summary)
    
    
    