# Chat with Documents (RAG System)

A local-first Retrieval-Augmented Generation (RAG) pipeline that lets you upload PDFs and DOCX files and ask natural-language questions about their content. Built to run entirely on local infrastructure using Ollama, with the option to deploy as a public REST API.

## Features

- **Document ingestion** — parses and sanitizes text from irregular PDF and DOCX formats, handling inconsistent layouts and encoding issues
- **Token-overlap chunking** — splits documents into overlapping segments to preserve context across chunk boundaries and reduce semantic loss during vectorization
- **Semantic search** — indexes embeddings in ChromaDB and retrieves relevant chunks via cosine similarity scoring
- **Local LLM inference** — uses Ollama to generate answers without sending data to external APIs
- **REST API** — all functionality exposed through FastAPI endpoints (upload, query, health check)
- **Rate limiting** — per-IP request throttling via SlowAPI to prevent abuse and keep the service stable under load
- **Containerized deployment** — packaged with Docker and deployed on an AWS EC2 instance

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| Vector store | ChromaDB |
| LLM runtime | Ollama |
| Orchestration | LangChain |
| Containerization | Docker |
| Deployment | AWS EC2 |
| Rate limiting | SlowAPI |

## Architecture

```
Client → FastAPI → [Ingestion Pipeline] → Chunking → Embedding → ChromaDB
                                                                      │
                          Client ← Answer ← Ollama (LLM) ← Retrieved Chunks
```

## Getting Started

### Prerequisites
- Python 3.10+
- Docker
- Ollama installed locally (or accessible endpoint)

### Installation

```bash
git clone https://github.com/InnocentPerson/chat-with-documents.git
cd chat-with-documents
pip install -r requirements.txt
```

### Running locally

```bash
uvicorn app.main:app --reload --port 8000
```

### Running with Docker

```bash
docker build -t chat-with-documents .
docker run -p 8000:8000 chat-with-documents
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a PDF/DOCX file for ingestion |
| `POST` | `/query` | Ask a question against ingested documents |
| `GET` | `/health` | Health check endpoint |

### Example request

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the termination clause in this contract?"}'
```

## Environment Variables

```
OLLAMA_HOST=http://localhost:11434
CHROMA_DB_PATH=./data/chroma
RATE_LIMIT=10/minute
```

## Roadmap

- [ ] Support for multi-file batch ingestion
- [ ] Streaming responses
- [ ] Frontend UI for non-technical users

## License

MIT
