PDF Chatbot — Chat with Your Documents

A simple AI-powered chatbot that allows users to upload PDF files, extract text, and ask questions about the content using Hugging Face and LangChain.
Built with Python, FastAPI / Streamlit, and FAISS for retrieval-augmented generation (RAG).

Features:

~Upload single or multiple PDF files
~Ask questions based on document content
~Uses Hugging Face or Google Generative AI for responses
~Vector database built with FAISS for fast retrieval
~Built with LangChain, Streamlit, and PyPDF2
~Simple chat-like interface with conversation history download

Technologies used:

Frontend-Streamlit (HTML/CSS/JS)
Backend-Logic	Python
Document Parsing-PyPDF2
Embeddings & LLM-Hugging Face / Google Generative AI
Vector Store-FAISS
Environment-Virtualenv
Persistence-CSV export (for chat history)

Installation & Setup
1.Clone this repository
git clone https://github.com/srihaasa27/pdf-chatbot.git
cd pdf-chatbot

2.Create and activate a virtual environment
python -m venv myenv

(On PowerShell):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\myenv\Scripts\activate

3.Install dependencies
pip install -r requirements.txt

4.Set up API keys:
Create a .env file or use Streamlit sidebar to enter:
GOOGLE_API_KEY

5.How to Run:
streamlit run app.py

Once it starts, open the URL shown in your terminal (usually http://localhost:8501).

6.How It Works:
Upload PDFs → The app reads and extracts text using PyPDF2.
Split text into chunks → Uses LangChain’s RecursiveCharacterTextSplitter.
Generate embeddings → With HuggingFaceEmbeddings.
Store in FAISS → For quick similarity search.
Ask questions → User query is matched to the most relevant text chunks.
LLM generates answer → Based on context retrieved from PDFs.