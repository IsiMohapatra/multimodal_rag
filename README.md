# Multimodal RAG

A Python-based **Multimodal Retrieval-Augmented Generation (RAG)** system for processing and querying documents with both text and images. This project integrates tools like LangChain, ChromaDB, and Docling to build a flexible RAG pipeline.

## Project Overview

This repository provides a multimodal RAG system capable of:

- Processing PDF and image documents.
- Converting documents into structured formats using Docling.
- Creating embeddings using OpenAI or other supported models.
- Storing and querying embeddings in ChromaDB.
- Integrating with LangChain for retrieval-augmented generation workflows.

---

## Steps to setup

1. **Clone the repository**
```bash
git clone https://github.com/IsiMohapatra/multimodal_rag.git
cd multimodal_rag

2. **Create the virtual env**
python -m venv multimodal_rag

3. **Activate virtual env**
multimodal_rag\Scripts\activate

4.**Install the dependencies**
pip install -r requirement.txt

How the Complete System Works:

1. Document Processing Pipeline:
PDF Upload → DocIngest → Docling Processing → DoclingLoader → Vector Store (Chroma)

•	rag_setup.py: Processes PDFs using Docling, extracts text, images, tables and then chunks them and store them in the ChromaDb database

2. RAG (Retrieval-Augmented Generation) Pipeline:

•	rag_test.py

