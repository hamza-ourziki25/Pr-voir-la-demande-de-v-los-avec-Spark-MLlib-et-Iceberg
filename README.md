# Assistant RAG Local — Ollama + ChromaDB + Streamlit

A fully local document assistant that answers questions using information extracted from your PDF files.

**Your data stays on your computer. No OpenAI API key required.**

Built with Python, Streamlit, LangChain, ChromaDB, and Ollama.

## Features

- **PDF Upload:** Upload any text-based PDF and index it with one click.
- **Semantic Search:** Find relevant passages using embedding similarity.
- **Local LLM:** Generate answers using Llama 3.2 through Ollama.
- **Context-Based Answers:** Answers are generated exclusively from the retrieved PDF context.
- **No Hallucinations (Prompt-Based):** If the answer is not found in the context, the assistant responds with `JE NE SAIS PAS`.
- **Source Transparency:** Display the exact passages used to generate each answer.
- **Prompt Inspection:** View the complete augmented prompt sent to the LLM.
- **SHA256 Deduplication:** Avoid re-indexing the same PDF multiple times.
- **Persistent Storage:** Store document embeddings locally using ChromaDB.
- **100% Local Execution:** No cloud services or external API keys required.

## How It Works — RAG Pipeline

The application follows a Retrieval-Augmented Generation (RAG) architecture:

1. **Upload:** The user uploads a PDF through the Streamlit interface.
2. **Load:** `PyPDFLoader` extracts the text page by page.
3. **Split:** `RecursiveCharacterTextSplitter` divides the extracted text into smaller chunks.
4. **Embed:** `nomic-embed-text` converts each chunk into a numerical vector.
5. **Store:** ChromaDB stores the embeddings and associated document metadata.
6. **Retrieve:** The user's question is embedded, and the two most relevant chunks are retrieved.
7. **Generate:** Llama 3.2 generates an answer using only the retrieved context.
8. **Display:** The application displays the answer, retrieved passages, and augmented prompt.

## Chunking Strategy

The application uses `RecursiveCharacterTextSplitter` with a tokenizer:

```python
RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=768,
    chunk_overlap=16,
)
```

### Configuration

- **chunk_size = 768:** Maximum chunk size of 768 tokens. This provides a balance between contextual information and retrieval precision.
- **chunk_overlap = 16:** Maintains 16 overlapping tokens between consecutive chunks to reduce information loss at chunk boundaries.
- **encoding_name = "cl100k_base":** Tokenizer used to estimate the number of tokens in each chunk.

*Note: Token counts are estimates based on the selected tokenizer and may differ from Ollama's tokenizer.*

## Prompt Template

The following prompt instructs the LLM to answer strictly using the retrieved context:

```text
Tu es un assistant qui répond à des questions sur un document PDF.

Réponds uniquement à partir du CONTEXTE ci-dessous.

Si la réponse ne se trouve pas dans le CONTEXTE,
réponds exactement : JE NE SAIS PAS.

N'invente rien.
Réponds en français, de façon claire et concise.
```

## Prerequisites

Before running the application, make sure you have:

- Python 3.10 or higher.
- Ollama installed: https://ollama.com
- At least 4 GB of RAM (8 GB recommended).
- Windows, Linux, or macOS.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install them manually:

```bash
pip install streamlit chromadb langchain-community langchain-ollama langchain-text-splitters pypdf tiktoken
```

### 4. Download Ollama models

Download the embedding model:

```bash
ollama pull nomic-embed-text
```

Download the language model:

```bash
ollama pull llama3.2:1b
```

### 5. Verify Ollama

Make sure Ollama is running and the models are available:

```bash
ollama list
```

If Ollama is not running, start it with:

```bash
ollama serve
```

## Usage

### 1. Start the application

```bash
streamlit run app.py
```

The application will be available at:

http://localhost:8501

### 2. Index a PDF

1. Open the sidebar.
2. Click **Choisir un PDF**.
3. Select a PDF document.
4. Click **Indexer le PDF**.
5. Wait for the confirmation message: `Indexation terminée !`.
6. Check the document statistics: file size, number of pages, and number of chunks.

### 3. Ask a question

1. Enter your question in the input field.
2. Click **Obtenir une réponse**.
3. Read the generated answer.
4. Open **Voir les passages récupérés** to inspect the source passages.
5. Open **Voir le prompt augmenté** to inspect the complete context sent to Llama.

### 4. Change the PDF

Simply upload another PDF document.

The application automatically resets the current document state and processes the newly uploaded file.

## Project Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── uploads_rag/
└── chroma_db_rag/
```

| File / Folder | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `README.md` | Project documentation |
| `requirements.txt` | Python dependencies |
| `uploads_rag/` | Stores uploaded PDF files |
| `chroma_db_rag/` | Persistent ChromaDB vector database |

## Configuration

The main application parameters are:

| Parameter | Value | Description |
|---|---|---|
| `UPLOAD_DIR` | `./uploads_rag` | PDF storage directory |
| `CHROMA_DIR` | `./chroma_db_rag` | ChromaDB storage directory |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `LLM_MODEL` | `llama3.2:1b` | Generative language model |
| `chunk_size` | `768` | Maximum chunk size in tokens |
| `chunk_overlap` | `16` | Overlapping tokens |
| `k` | `2` | Number of retrieved chunks |
| `temperature` | `0` | Reduces randomness in generation |

### Using a Larger Language Model

For potentially better responses on complex questions, you can use:

```python
LLM_MODEL = "llama3.2:3b"
```

This model requires more memory and computational resources.

## Key Functions

The application is organized around three main functions:

### `rechercher_passages()`

Retrieves the `k` most relevant chunks from ChromaDB based on the semantic similarity between the user's question and stored document embeddings.

### `construire_prompt()`

Combines the retrieved passages with the user's question to construct the augmented prompt sent to the language model.

### `generer_reponse()`

Builds the final prompt, calls Llama 3.2 through Ollama, and returns the generated answer.

## Troubleshooting

| Problem | Possible Cause | Solution |
|---|---|---|
| No text extracted | The PDF is scanned or contains images | Apply OCR before indexing |
| Connection refused | Ollama is not running | Start Ollama using `ollama serve` |
| Always returns `JE NE SAIS PAS` | Retrieved context does not contain enough information | Increase `k` from 2 to 4 or reduce `chunk_size` |
| Model not found | Required Ollama model is missing | Run `ollama pull MODEL_NAME` |
| Slow responses | Limited hardware resources | Use a smaller model or reduce the context size |

## Limitations

- Supports text-based PDFs only; scanned documents require OCR.
- Llama 3.2:1b is lightweight but has limited capabilities for complex reasoning.
- No conversation history is currently implemented.
- Only one question is processed at a time.
- Answer quality depends on document quality, chunking, retrieval accuracy, and the selected language model.
- The prompt encourages context-grounded answers but does not guarantee that the model will never generate unsupported information.

## Future Improvements

- Add OCR support for scanned PDFs.
- Implement conversation history and multi-turn interactions.
- Support multiple PDF documents simultaneously.
- Add a similarity score threshold to filter irrelevant passages.
- Implement document metadata filtering.
- Experiment with different embedding models and retrieval strategies.
- Add a reranking step to improve retrieval quality.

## License

This project is licensed under the MIT License.

Free to use, modify, and distribute for educational and personal projects.