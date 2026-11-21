# Multimodal Agentic RAG System

A multimodal Retrieval-Augmented Generation (RAG) application that combines text and image retrieval with an intelligent AI agent.

## Project Overview

This project implements a multimodal RAG pipeline based on the research paper **"Attention Is All You Need"**.

It enables users to retrieve relevant information from both textual and visual content using CLIP embeddings, ChromaDB vector storage, and a LangGraph-powered agent.

## Key Features

- **PDF Extraction:** Extract textual content and images from PDF documents.
- **Multimodal Embeddings:** Generate text and image embeddings using OpenAI CLIP.
- **Vector Database:** Store and retrieve embeddings using ChromaDB.
- **Semantic Search:** Retrieve relevant content using text or image queries.
- **AI Agent:** Integrate retrieval tools into a LangGraph agent.
- **ReAct Architecture:** Allow the agent to select and execute the appropriate retrieval tool.

## Architecture

```text
                 PDF Document
                      |
              PDF Extraction
                      |
              +-------+-------+
              |               |
          Text Data       Image Data
              |               |
          CLIP Text       CLIP Image
          Embeddings      Embeddings
              |               |
              +-------+-------+
                      |
                  ChromaDB
                      |
              Retrieval Tools
                      |
                LangGraph Agent
                      |
                AI Response
```

## Technologies

| Category | Technologies |
|---|---|
| Programming Language | Python |
| PDF Processing | PyMuPDF |
| Image Processing | Pillow |
| Multimodal AI | OpenCLIP, PyTorch |
| Vector Database | ChromaDB |
| Agent Framework | LangChain, LangGraph |
| LLM Integration | LangChain OpenAI |
| Environment Management | uv |

## Project Structure

```text
rag-multimodal/
├── data/
├── images/
├── chroma_db/
├── src/
│   ├── extract.py
│   ├── embeddings.py
│   ├── index.py
│   ├── retrieval.py
│   ├── retriever_tool.py
│   ├── agent.py
│   ├── test_retrieval.py
│   └── test_tools.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-multimodal.git
cd rag-multimodal
```

### 2. Install dependencies

Install [uv](https://docs.astral.sh/uv/) if it is not already installed.

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file based on `.env.example`.

```bash
OPENAI_API_KEY=your_openai_api_key
```

Never commit your actual API key.

### 4. Add the PDF document

Place the research paper in the `data/` directory:

```text
data/Attention is all you need.pdf
```

### 5. Run the pipeline

Execute the scripts in the following order:

```bash
uv run python src/extract.py
uv run python src/embeddings.py
uv run python src/index.py
```

### 6. Test retrieval

```bash
uv run python src/test_retrieval.py
uv run python src/test_tools.py
```

### 7. Run the AI agent

```bash
uv run python src/agent.py
```

An OpenAI API key with available API credits is required for the OpenAI-powered agent configuration.

## Retrieval Examples

**Text query:**

```text
What is the Transformer architecture?
```

**Image query:**

```text
images/page_3_image_1.png
```

The retrieval system uses CLIP embeddings to search for semantically similar content in the vector database.

## Future Improvements

- Support user-uploaded PDF documents.
- Improve text chunking and retrieval quality.
- Add a web interface using Streamlit or FastAPI.
- Support local LLMs for offline inference.
- Add evaluation metrics for retrieval performance.
- Implement multimodal conversational interactions.

## Research Reference

Vaswani et al. (2017).  
*Attention Is All You Need.*

[Read the research paper on arXiv](https://arxiv.org/abs/1706.03762).

## Author

**Najoua Mouaddab**

Engineering Student | BigData | Artificial Intelligence

Interested in Artificial Intelligence, Data Science, and intelligent applications.

---

*Developed as an academic project exploring multimodal retrieval and agentic RAG architectures.*