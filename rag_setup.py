# setup.py
import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

from config import PDF_FOLDER, MARKDOWN_DIR, CHUNKS_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ImageRefMode

import chromadb
from chromadb.utils import embedding_functions
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    """Make sure all output directories exist before running pipeline."""
    for d in [MARKDOWN_DIR, CHUNKS_DIR, CHROMA_DB_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"✅ Ensured directories: {MARKDOWN_DIR}, {CHUNKS_DIR}, {CHROMA_DB_DIR}")


# -----------------------------
# 1️⃣ PDF Parsing with Docling
# -----------------------------
def convert_with_image_annotation(input_doc_path):
    if not api_key:
        raise RuntimeError("❌ OPENAI_API_KEY not found. Please set it in .env")

    model = "gpt-4.1-mini"
    picture_desc_api_option = PictureDescriptionApiOptions(
        url="https://api.openai.com/v1/chat/completions",
        prompt="Describe this image in sentences in a single paragraph.",
        params=dict(model=model),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        do_picture_description=True,
        picture_description_options=picture_desc_api_option,
        generate_page_images=True,
        enable_remote_services=True,
        generate_picture_images=True,
        images_scale=2,
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    conv_res = converter.convert(source=input_doc_path)
    return conv_res

def export_single_md_with_images_and_serials(conv_res, output_path: Path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem.replace(" ", "_")
    md_filename = output_path / f"{doc_filename}-full-with-serials.md"

    conv_res.document.save_as_markdown(
        md_filename,
        image_mode=ImageRefMode.REFERENCED,
        include_annotations=True,
        page_break_placeholder="<!-- PAGE_BREAK -->",
    )

    # Add page-end markers
    with open(md_filename, "r", encoding="utf-8") as f:
        md_text = f.read()

    pages = md_text.split("<!-- PAGE_BREAK -->")
    final_md = ""
    for idx, page_text in enumerate(pages, start=1):
        page_text = page_text.strip()
        if not page_text:
            continue
        final_md += page_text + f"\n\n<!-- PAGE {idx} END -->\n\n"

    Path(md_filename).write_text(final_md, encoding="utf-8")
    print(f"✅ Markdown saved with serials → {md_filename}")
    return md_filename

def parse_pdfs():
    print("📄 Parsing PDFs with Docling pipeline...")
    markdown_files = []
    pdf_dir = Path(PDF_FOLDER)  # ✅ convert str → Path
    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"   ➡ Converting {pdf_path}")
        conv_res = convert_with_image_annotation(pdf_path)
        md_file = export_single_md_with_images_and_serials(conv_res, MARKDOWN_DIR)
        markdown_files.append(md_file)
    return markdown_files

# -----------------------------
# 2️⃣ Hybrid Chunking
# -----------------------------
def chunk_markdowns(markdown_files, max_tokens=1000):
    all_chunks = []
    for md_file in markdown_files:
        md_file = Path(md_file)
        with md_file.open("r", encoding="utf-8") as f:
            md_text = f.read()

        # Clean up paragraphs
        lines = md_text.split("\n")
        buffer, cleaned_text = [], ""
        for line in lines:
            buffer.append(line)
            if line.strip() == "":
                cleaned_text += "\n".join(buffer) + "\n"
                buffer = []
        if buffer:
            cleaned_text += "\n".join(buffer)

        # Split by headings
        sections = re.split(r"(#+ .+)", cleaned_text)

        chunks, current_chunk, current_pages, current_heading = [], "", set(), ""
        page_marker_pattern = re.compile(r"<!-- PAGE (\d+) END -->")

        for sec in sections:
            if not sec.strip():
                continue

            # heading
            heading_match = re.match(r"(#+) (.+)", sec)
            if heading_match:
                current_heading = heading_match.group(2)

            # pages
            pages_in_sec = page_marker_pattern.findall(sec)
            for p in pages_in_sec:
                current_pages.add(int(p))

            if len(current_chunk + sec) <= max_tokens:
                current_chunk += sec + "\n"
            else:
                if current_chunk:
                    chunks.append({
                        "pages": sorted(current_pages) if current_pages else [1],
                        "heading": current_heading,
                        "chunk_text": current_chunk.strip()
                    })
                current_chunk = sec + "\n"
                current_pages = set(int(p) for p in pages_in_sec)

        if current_chunk:
            chunks.append({
                "pages": sorted(current_pages) if current_pages else [1],
                "heading": current_heading,
                "chunk_text": current_chunk.strip()
            })

        # Save JSON
        json_path = Path(CHUNKS_DIR) / f"{md_file.stem}_chunks.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)

        print(f"✅ {len(chunks)} chunks saved → {json_path}")
        all_chunks.extend(chunks)
    return all_chunks

# -----------------------------
# 3️⃣ Build Embeddings + Chroma
# -----------------------------

def build_embeddings(chunks):

    print("🔎 Building embeddings with OpenAIEmbeddings + Chroma...")

    # Initialize Chroma client
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    collection_name = "pdf_chunks"
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"🗑 Deleted existing collection '{collection_name}'")

    # Create new collection with OpenAI embedding function
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
    )

    ids = [f"chunk_{i+1}" for i in range(len(chunks))]
    texts = [c["chunk_text"] for c in chunks]
    metadatas = [{"pages": ",".join(map(str, c.get("pages", []))), "heading": c.get("heading", "")} for c in chunks]

    # Compute embeddings with OpenAIEmbeddings (debug print)
    embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)
    embeddings = embeddings_model.embed_documents(texts)
    for i, emb in enumerate(embeddings):
        print(f"Chunk {i+1} embedding length: {len(emb)}")

    # Add documents to Chroma
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB @ {CHROMA_DB_DIR}")
    print(f"📚 Collection '{collection_name}' now contains {len(collection.get()['ids'])} documents")

    return collection


# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()
    ensure_dirs()
    markdowns = parse_pdfs()
    chunks = chunk_markdowns(markdowns)
    build_embeddings(chunks)
    print("\n🎉 Setup complete! Multimodal RAG is ready.")

if __name__ == "__main__":
    main()
