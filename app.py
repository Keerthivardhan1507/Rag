from src.data_loader import read_all_documents
import os


if __name__ =="__main__":
    docs = read_all_documents("data")
    print(docs)
    
    
    