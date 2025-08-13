from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import tempfile
import os
from typing import List, Optional
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document from file path"""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding='utf-8')
        
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_from_upload(self, uploaded_file) -> List[Document]:
        """Load document from uploaded file"""
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            chunks = self.load_document(tmp_file_path)
            return chunks
        finally:
            os.unlink(tmp_file_path)
    
    def create_vectorstore(self, chunks: List[Document]) -> FAISS:
        """Create FAISS vector store from chunks"""
        return FAISS.from_documents(chunks, self.embeddings)
    
    def search_documents(self, vectorstore: FAISS, query: str, k: int = 3) -> List[Document]:
        """Search for relevant documents"""
        return vectorstore.similarity_search(query, k=k)