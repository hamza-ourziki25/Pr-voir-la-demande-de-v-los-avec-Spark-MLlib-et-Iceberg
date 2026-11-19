# Assistant RAG Local - Ollama + ChromaDB + Streamlit

A local document assistant that answers questions only from your PDF.
No data leaves your computer. No OpenAI key needed.

Built with: Python, Streamlit, LangChain, ChromaDB, Ollama (Llama 3.2 + Nomic Embed)

## Features
- Upload any PDF and index it in one click
- Ask questions in natural language
- Answers ONLY from the PDF context. If not found, it says: JE NE SAIS PAS
- Shows the exact passages used + the full augmented prompt
- No re-indexing if the same PDF is uploaded again (SHA256 deduplication)
- Persistent vector database with ChromaDB
- Full local execution with Ollama

## How It Works (RAG Pipeline)
1. Upload - User uploads a PDF in Streamlit
2. Load - PyPDFLoader extracts text, page by page
3. Split - Text is split into chunks of 768 tokens, overlap 16
4. Embed - Chunks are embedded with nomic-embed-text
5. Store - Embeddings are stored in ChromaDB
6. Retrieve - Question is embedded and top 2 closest chunks are retrieved
7. Generate - Llama 3.2 generates an answer from the context only
8. Display - Answer + sources + prompt are displayed

### Chunking Strategy
```python
RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=768,
    chunk_overlap=16,
)
- chunk_size=768: Max size of one chunk, about 550 words / 1 page. Balanced between precision and context.
- chunk_overlap=16: Number of tokens repeated between chunks, to avoid cutting a sentence in the middle.
- encoding_name="cl100k_base": Tokenizer used to count tokens correctly.

### Prompt Template
Tu es un assistant qui répond à des questions sur un document PDF.
Réponds uniquement à partir du CONTEXTE ci-dessous.
Si la réponse ne se trouve pas dans le CONTEXTE, réponds exactement : JE NE SAIS PAS.
N'invente rien. Réponds en français, de façon claire et concise.
## Prerequisites
- Python 3.10+
- Ollama installed: https://ollama.com
- At least 4 GB RAM (8 GB recommended)

## Installation
### 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate
### 2. Install dependencies
pip install streamlit chromadb langchain-community langchain-ollama langchain-text-splitters pypdf
### 3. Pull Ollama models
ollama pull nomic-embed-text
ollama pull llama3.2:1b
Check Ollama is running:
ollama list
## Usage
### 1. Start the app
streamlit run app.py
The app opens at: http://localhost:8501

### 2. Index a PDF
1. In the sidebar, click Choisir un PDF
2. Select a PDF file
3. Click Indexer le PDF
4. Wait for: Indexation terminée !
5. You will see: Size, Pages, Chunks

### 3. Ask a question
1. Type your question
2. Click Obtenir une réponse
3. Read the answer
4. Open Voir les passages récupérés to verify the source
5. Open Voir le prompt augmenté to see what was sent to Llama

### 4. Change PDF
Just upload a new PDF. The old state is automatically reset.

## Project Structure
.
├── app.py
├── README.md
├── uploads_rag/
├── chroma_db_rag/
└── requirements.txt
## Configuration
- UPLOAD_DIR = ./uploads_rag
- CHROMA_DIR = ./chroma_db_rag
- EMBED_MODEL = nomic-embed-text
- LLM_MODEL = llama3.2:1b
- chunk_size = 768
- chunk_overlap = 16
- k = 2
- temperature = 0

To use a bigger model:
LLM_MODEL = "llama3.2:3b"
## Key Functions
- rechercher_passages(): Retrieves the k closest chunks to the question
- construire_prompt(): Joins passages into a context and fills the prompt template
- generer_reponse(): Builds the prompt and calls Llama to generate the answer

## Troubleshooting
- Aucun texte extrait: PDF scanné ? -> Your PDF is an image. Use OCR first.
- Connection refused -> Start Ollama: ollama serve
- Answer is always JE NE SAIS PAS -> Try increasing k from 2 to 4, or decreasing chunk_size to 500.

## Limitations
- Works only with text-based PDFs, not scanned images (no OCR yet)
- Llama 3.2:1b is fast but limited for complex reasoning
- No chat history, one question at a time

## License
MIT - Free to use for learning and personal projects.