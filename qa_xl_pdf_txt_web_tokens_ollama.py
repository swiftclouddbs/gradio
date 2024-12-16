import gradio as gr
import langchain
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.llms import Ollama
import requests
from bs4 import BeautifulSoup

def process_query(query, pdf_file, text_file, excel_file, url):

    text = ""  # Initialize an empty string for text
    
    if excel_file:
        excel_text = load_excel(excel_file)
        text = excel_text
    elif pdf_file:
        pdf_text = load_pdf(pdf_file)
        text = pdf_text
    elif text_file:
        text = load_text(text_file)
    elif url:
        text = fetch_text_from_url(url)
    else:
        prompt = query

    model = Ollama(model="llama3.1")  # Replace with your Ollama model name

    # Construct the prompt, incorporating the query
    prompt = f"Here is the document:\n{text}\n\nQuestion: {query}\n\nAnswer:"

    response = model(prompt)
    return response, f"Tokens Used: {len(response.split())}"  # Estimate token usage

def load_excel(excel_file):
    loader = UnstructuredExcelLoader(excel_file)
    documents = loader.load()
    excel_text = "\n".join([doc.page_content for doc in documents])
    return excel_text

def load_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    pdf_text = "\n".join([doc.page_content for doc in documents])
    return pdf_text

def load_text(text_file):
    with open(text_file, 'r') as f:
        text = f.read()
    return text

def fetch_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = " ".join(str(node) for node in soup.find_all(string=True))
    return text

iface = gr.Interface(
    fn=process_query,
    inputs=[
        gr.Textbox(label="Your Question"),
        gr.File(label="Upload PDF (optional)"),
        gr.File(label="Upload Text File (optional)"),
        gr.File(label="Upload Excel File (optional)"),
        gr.Textbox(label="Enter URL (optional)")
    ],
    outputs=[
        gr.Textbox(label="Response"),
        gr.Textbox(label="Token Usage")
    ],
    title="AI Inference for Excel, PDF, Text & Web Pages with Token Count",
    description="<div style='text-align: center;'>by Reggie Stuart, 691950170reg@gmail.com</div>",
)

iface.launch(server_name="0.0.0.0", server_port=7860)
