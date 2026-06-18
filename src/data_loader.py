import os
from langchain_community.document_loaders import  PyPDFLoader,TextLoader
from typing import List,Any
from pathlib import Path



def read_all_documents(pdf_directory:str) -> List[str]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    documents = []
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        print(f"Loading pdf {pdf_file}")
        try:
            loader = PyPDFLoader(pdf_file)
            loaded = loader.load()
            documents.extend(loaded)
        except Exception as e:
            print(f"error while loading the model{e}")
    return documents

    
        
    
    
