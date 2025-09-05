# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
# from langchain.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings


# Extract text from PDF
def load_pdf_files(data):
    loader =DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
        Given a list Document objects, return a new list of Document objects 
        containing only 'source','page' in metadata and the original 'page_content'.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        page = doc.metadata.get("page")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src, "page": page}
            )
        )

    return minimal_docs

# Spllit the docs into smaller chunks
def text_split(minimal_docs): 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(minimal_docs)

    return text_chunks

# # Download the embedding model from HuggingFace
# def download_hugging_face_embeddings():
#     """
#     Download and return the HuggingFace embeddings model.
#     """

#     model_name = "sentence-transformers/all-MiniLM-L6-v2"
#     embeddings = HuggingFaceEmbeddings(model_name= model_name)

#     return embeddings


# src/helper.py

# ❌ OLD (deprecated):
# from langchain.embeddings import HuggingFaceEmbeddings

# ✅ NEW (updated):
from langchain_huggingface import HuggingFaceEmbeddings

def download_hugging_face_embeddings():
    """
    Download and return HuggingFace embeddings model
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print(f"✅ Successfully loaded embeddings model: {model_name}")
        return embeddings
    except Exception as e:
        print(f"❌ Error loading embeddings model: {e}")
        # Fallback to a different model if needed
        fallback_model = "sentence-transformers/all-mpnet-base-v2"
        try:
            embeddings = HuggingFaceEmbeddings(model_name=fallback_model)
            print(f"✅ Loaded fallback embeddings model: {fallback_model}")
            return embeddings
        except Exception as e2:
            print(f"❌ Error with fallback model: {e2}")
            raise e2