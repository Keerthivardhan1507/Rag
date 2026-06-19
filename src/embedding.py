from src.data_loader import read_all_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import os
from typing import List,Any
from sentence_transformers import SentenceTransformer
import numpy as np

class embeddingPipeline:
    def __init__(self,model_name = "all-MiniLM-L6-v2",chunk_size :int =1000,chunk_overlap:int= 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] loading embedding model{self.model}" )
        
    def chunk_documents(self,documents:List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators=["\n\n","\n"," ",""]
        )
        
        chunks = splitter.split_documents(documents)
        print(f"Split  documnets {len(documents)} into number of {len(chunks)}chunks")
        return chunks
    
    def embed_chunks(self,chunks:List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO]generating embeddings for {len(texts)}")
        embeddings = self.model.encode(texts,show_progress_bar=True)
        print(f"[INFO]The generated embeddings{len(embeddings)}")
        return embeddings
    
if __name__ == "__main__":
    
    docs = read_all_documents("data")
    emb_pipeline = embeddingPipeline()
    chunks = emb_pipeline.chunk_documents(docs)
    embeddings = emb_pipeline.embed_chunks(chunks)
    print("[INFO]examble embedding:",embeddings[0] if len(embeddings) >0 else None )
    
