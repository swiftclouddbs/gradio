#Script needs to run in same directory as vectorstore

import gradio as gr
import chromadb
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# ... (same code as before)

def process_query(query):
    result = qa(query)
    return result

iface = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(lines=2, label="Enter your query"),
    outputs=gr.Textbox(lines=5, label="Response"),
    title="Ask a Question About America",
    description="Ask me anything about America!"
)

# Connect to the ChromaDB client
client = chromadb.Client()  # Replace with your ChromaDB path

# Create embeddings
embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

# Load the Chroma collection
collection_name = "America"

# Specify the directory where the Chroma vector data is stored
PERSIST_DIRECTORY = "./persist_nov_22_b"  # Update with your directory path

# Connect to your Chroma database (vector store) with persistence
vectorstore = Chroma(
    collection_name=collection_name, 
    embedding_function=embedding_model,
    persist_directory=PERSIST_DIRECTORY  # Specify persistent directory
)

# Create the LLM
llm = Ollama(model="llama3.1")

# Create the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(), 
#    chain_type_kwargs={"prompt": prompt},
    verbose=True
)

def process_query(query):
    result = qa(query)
    return result

iface = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(lines=2, label="Enter your query"),
    outputs=gr.Textbox(lines=5, label="Response"),
    title="Ask a Question About America",
    description="Ask me anything about America!"
)

iface.launch(server_name="0.0.0.0", server_port=7860)
