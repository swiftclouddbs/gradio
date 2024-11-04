import gradio as gr
import langchain
from langchain.document_loaders import PyPDFLoader, TextLoader
import google.generativeai as ai
import requests
from bs4 import BeautifulSoup

def process_query(query, pdf_file, text_file, url):
    if pdf_file:
        pdf_text = load_pdf(pdf_file)
        prompt = query + "\n" + pdf_text
    elif text_file:
        text = load_text(text_file)
        prompt = query + "\n" + text
    elif url:
        text = fetch_text_from_url(url)
        prompt = query + "\n" + text
    else:
        prompt = query

    ai.configure(api_key="AIzaSyAUvJhcDh9ZkdQbVNf4alQl5ZKVWfJsKtw")
    model = ai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    token_usage = model.count_tokens(prompt)
    return response.text, f"Tokens Used: {token_usage}"

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
        gr.Textbox(label="Enter URL (optional)")
    ],
    outputs=[
        gr.Textbox(label="Response"),
        gr.Textbox(label="Token Usage")
    ],
    title="Document Query with Token Count",
    description="This program uses Gemini to get the answers to your questions from documents or URLs"
)

iface.launch(server_name="0.0.0.0", server_port=7860)
