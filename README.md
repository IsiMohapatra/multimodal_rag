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

