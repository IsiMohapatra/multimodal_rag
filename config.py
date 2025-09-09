# config.py

# Paths
PDF_FOLDER = "docs/pdf_source"           # Folder containing PDFs
OUTPUT_DIR = "output"                    # All outputs go here
MARKDOWN_DIR = f"{OUTPUT_DIR}/markdowns"
CHUNKS_DIR = f"{OUTPUT_DIR}/chunks"

CHROMA_DB_DIR = f"{OUTPUT_DIR}/chroma_db"

# Models
EMBEDDING_MODEL = "open-ai embeddings"
LLM_MODEL = "gpt-4o-mini"
