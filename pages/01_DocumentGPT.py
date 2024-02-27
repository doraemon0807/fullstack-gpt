import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore

st.set_page_config(page_title="DocumentGPT", page_icon="🤖")


def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    # Open the file
    with open(file_path, "wb") as f:
        # Saved the opened file into file_path
        f.write(file_content)

    # directory for saving embeddings
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    # spliting the file into chunks
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader("./files/chapter_one.txt")
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()
    return retriever


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
"""
)

# Create a file uploader
file = st.file_uploader(
    "Upload a .txt, .pdf, or .docx file.", type=["pdf", "txt", "docx"]
)

if file:
    retriever = embed_file(file)
    s = retriever.invoke("winston")
    s
