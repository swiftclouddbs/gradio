#Updated gradio interface for RAG
#Need to create complimentary app to load documents
#Need to alter hard-coded collection_name & PERSIST_DIRECTORY
#

import chromadb
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# Prompt Template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Connect to the ChromaDB client
client = chromadb.Client()

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
llm = OllamaLLM(model="llama3")

# Define the QA chain
qa_chain = (
    {
        "context": vectorstore.as_retriever(),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Function to process the input and retrieve the answer
def get_answer(question):
    result = qa_chain.invoke(question)
    return result

# Create the Gradio Interface
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Question Answering System with Context Retrieval")
        with gr.Row():
            question_input = gr.Textbox(label="Ask a Question", placeholder="Type your question here...")
            answer_output = gr.Textbox(label="Answer", interactive=False)
        submit_button = gr.Button("Submit Question")

        submit_button.click(fn=get_answer, inputs=question_input, outputs=answer_output)
        
    return demo

# Launch the Gradio interface
demo = create_gradio_interface()
demo.launch(server_name="0.0.0.0", server_port=7861)
