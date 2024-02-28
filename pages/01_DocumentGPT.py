import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="DocumentGPT", page_icon="🤖")

llm = ChatOpenAI(
    temperature=0.1,
)


# if the file didn't change, this function will not run again
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    # Open the file
    with open(file_path, "wb") as f:
        # Saved the opened file into cache directory
        f.write(file_content)

    # directory for saving embeddings
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    # spliting the file into chunks
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )

    # load the splitted file
    loader = UnstructuredFileLoader(file_path)

    # load the document
    docs = loader.load_and_split(text_splitter=splitter)

    # embed the document
    embeddings = OpenAIEmbeddings()

    # cache the embedding
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # put the embedding into vector store
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        # save new message by appending it to message state
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    # joining every document's page_content in docs with two line breaks
    return "\n\n".join(document.page_content for document in docs)


# Creating the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your file from the sidebar.
"""
)

# Create a file uploader
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt, .pdf, or .docx file.", type=["pdf", "txt", "docx"]
    )

# Run a function that embeds the file
if file:
    retriever = embed_file(file)

    send_message("I'm ready! Ask away!", "AI", save=False)

    # paint previously sent messages
    paint_history()

    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")

        # create a chain: use user's message as an input -> format the doc -> format the prompt -> give formatted prompt to llm
        chain = (
            {
                "context": retriever
                | RunnableLambda(
                    format_docs
                ),  # calls retriever(message) -> returns bunch of documents as an output -> calls format_docs(output) -> returns a chunk of joined string
                "question": RunnablePassthrough(),  # RunnablePassthrough: relays message
            }
            | prompt  # context and question from the above steps is given to prompt as an input
            | llm  # prompt from the above step is given to llm as an input
        )

        response = chain.invoke(message)
        send_message(response.content, "AI")


else:
    # initialize message state if no file is present
    st.session_state["messages"] = []
