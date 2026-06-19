import os
from dotenv import load_dotenv
from src.vector_store import vectorstore
from langchain_groq import ChatGroq

load_dotenv()

class RAGsearch:
    def __init__(self,persist_dir = "faiss_store",embedding_model = "all-MiniLM-L6-v2",llm_model = "llama-3.3-70b-versatile"):
        self.vectorstore = vectorstore(persist_dir,embedding_model)
        ## Load or build vector store
        faiss_path = os.path.join(persist_dir,"faiss.index")
        meta_path = os.path.join(persist_dir,"metadata.pkl")
        if not(os.path.exists(faiss_path) and (os.path.exists(meta_path))):
            from src.data_loader import read_all_documents
            docs = read_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = ""
        self.llm = ChatGroq(groq_api_key = groq_api_key,model_name = llm_model)
        print(f"[INFO] groq llm initializzed {llm_model}")
        
    def search_and_summarize(self,query:str,top_k:int = 5) -> str:
        results = self.vectorstore.query(query,top_k)
        texts=[r["metadata"].get("text","") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No Relavent documents found"
        prompt  = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content
    
if __name__ == "__main__":
    rag_search = RAGsearch()
    query = "What is EDA"
    summary = rag_search.search_and_summarize(query,top_k=3)
    print("summary:",summary)
            
     
