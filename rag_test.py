import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import spacy
import re
from docx import Document
from docx.shared import Inches

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Load ENV
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("❌ No OpenAI API key found in .env")
    sys.exit(1)

# -------------------------
# Paths
# -------------------------
CHROMA_DB_DIR = Path("output/chroma_db")
if not CHROMA_DB_DIR.exists() or not any(CHROMA_DB_DIR.iterdir()):
    logger.error("❌ Chroma DB not found. Please run RAG setup first.")
    sys.exit(1)

# -------------------------
# Semantic Retriever
# -------------------------
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(
    persist_directory=str(CHROMA_DB_DIR),
    embedding_function=embeddings,
    collection_name="pdf_chunks"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -------------------------
# LLM
# -------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY
)

# -------------------------
# Assistant Class
# -------------------------

def is_dtc_code(question: str) -> bool:
        """
        Detect if the query contains an OBD-II DTC code like P0301, U0100, etc.
        """
        return bool(re.search(r"\b([PBUC]\d{4})\b", question.upper()))


class Assistant:
    def __init__(self):
        self.retriever = retriever
        self.nlp = spacy.load("en_core_web_trf")
        self.memory = {
            "conversation_history": [],  # stores dicts with question/answer
            "last_topic": None,          # e.g., DTC code or vehicle info
            "vehicle_info": None,
            "dtc_code": None
        }

    def is_vehicle_related(self, question: str) -> bool:
        classifier_prompt = f"""
You are a classifier. Decide if the user question is about **vehicle diagnostics, repair, or automotive problems**.

QUESTION: {question}

Answer only with "YES" or "NO".
"""     
        print("it is a vehicle related problem")
        response = llm.predict(classifier_prompt).strip().upper()
        return response == "YES"

    def extract_vehicle_model(self, question: str) -> str:
        doc = self.nlp(question)
        entities = []

        # Step 1: spaCy NER extraction
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                model_tokens = [ent.text]
                next_token = ent.end
                while next_token < len(doc) and (
                    doc[next_token].is_title or doc[next_token].like_num or doc[next_token].is_lower
                ):
                    model_tokens.append(doc[next_token].text)
                    next_token += 1
                entities.append(" ".join(model_tokens))

        if entities:
            return entities[0].title()

        tokens = [t for t in doc if not t.is_stop and t.pos_ in ["PROPN", "NUM", "NOUN"]]
        if len(tokens) >= 2:
            # Join consecutive proper nouns/numbers until a stopword or verb
            model_tokens = []
            for t in tokens:
                if t.pos_ in ["PROPN", "NUM"]:
                    model_tokens.append(t.text)
                else:
                    break
            if model_tokens:
                return " ".join(model_tokens).title()

        return None



    def ask(self, question: str):
        try:
            # 🔹 Step 0: Check if query contains a DTC code first
            dtc_match = re.search(r"\b([PBUC]\d{4})\b", question.upper())
            if dtc_match:
                dtc_code = dtc_match.group(1)
                self.memory["dtc_code"] = dtc_code
                self.memory["last_topic"] = f"DTC {dtc_code}"
                # Directly send to LLM without asking for vehicle info
                final_question = question
            else:
                # 🔹 Step 1: If we are waiting for vehicle info, treat this input as that
                if "pending_question" in self.memory:
                    vehicle_info = self.extract_vehicle_model(question)
                    if vehicle_info:
                        merged_question = f"{self.memory['pending_question']} for {vehicle_info}"
                        self.memory["vehicle_info"] = vehicle_info
                        del self.memory["pending_question"]
                        final_question = merged_question
                    else:
                        return {
                            "answer": "I didn’t catch the make and model, can you repeat it clearly?",
                            "source_documents": []
                        }
                else:
                    # 🔹 Step 2: Normal vehicle-related check
                    if not self.is_vehicle_related(question):
                        return {
                            "answer": "Please ask me about vehicle-related problems, I’ll be happy to assist you.",
                            "source_documents": []
                        }

                    # 🔹 Step 3: Extract vehicle info
                    vehicle_info = self.extract_vehicle_model(question)
                    if not vehicle_info and not self.memory.get("vehicle_info"):
                        self.memory["pending_question"] = question
                        return {
                            "answer": "Can you please specify the make and model of your vehicle?",
                            "source_documents": []
                        }

                    if vehicle_info:
                        self.memory["vehicle_info"] = vehicle_info
                        self.memory["last_topic"] = vehicle_info
                        if "pending_question" in self.memory:
                            del self.memory["pending_question"]
                        final_question = question
                    elif self.memory.get("vehicle_info") and self.memory.get("pending_question"):
                        final_question = f"{question} for {self.memory['vehicle_info']}"
                    else:
                        final_question = question

            print(f"\n🚀 Final query sent to retriever/LLM: {final_question}\n")

            # 🔹 Step 4: Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(final_question)
            if not docs:
                return {"answer": "No relevant information found in the PDF.", "source_documents": []}

            # 🔹 Step 5: Build chunks
            context_text, chunk_mapping = "", {}
            for i, d in enumerate(docs, 1):
                label = f"CHUNK_{i}"
                context_text += f"{label}:\n{d.page_content}\n\n"
                chunk_mapping[label] = {
                    "page_number": d.metadata.get("pages", "N/A"),
                    "content": d.page_content
                }

            # 🔹 Step 6: Use history
            history_text = ""
            for turn in self.memory["conversation_history"][-5:]:
                history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

            # 🔹 Step 7: Build LLM prompt
            prompt = f"""
    You are a vehicle diagnostic assistant.
    Use the conversation history and provided chunks to answer the question.

    Conversation History:
    {history_text}

    QUESTION: {final_question}

    CHUNKS:
    {context_text}

    If NONE of these chunks fully contain the answer, respond ONLY with "NONE".
    Do not guess or partially answer. Your response must be either "NONE" or a chunk label like "CHUNK_3".
    """
            response = llm.invoke(prompt)  # returns AIMessage
            selected_label = response.content.strip().upper()

            # 🔹 Step 8: Select final answer
            if selected_label == "NONE" or selected_label not in chunk_mapping:
                answer_text = "No relevant information found in the PDF."
            else:
                answer_text = chunk_mapping[selected_label]["content"]

            # 🔹 Step 9: Save Q&A
            self.memory["conversation_history"].append({
                "question": final_question,
                "answer": answer_text
            })

            # 🔹 Step 10: Return with source
            source = []
            if selected_label in chunk_mapping:
                source = [{"page_number": chunk_mapping[selected_label]["page_number"]}]

            return {
                "answer": answer_text,
                "source_documents": source
            }

        except Exception as e:
            print(f"❌ Error in ask(): {e}")
            return {"answer": "Error occurred.", "source_documents": []}




# ---------------------------------------
# Save the conversation of the session
# ----------------------------------------

def save_conversation_to_word(conversation_history, output_path="Allion_Session.docx"):
    """
    Save conversation to Word and embed images.
    Dynamically finds the project root (multimodal_rag) and preserves
    the duplicated 'output/markdowns/output/markdowns' structure for images.
    """
    doc = Document()
    doc.add_heading("Allion Automotive Assistant Session", level=0)

    # 🔹 Dynamically detect multimodal_rag root folder
    project_root = Path(__file__).resolve()
    while project_root.name != "multimodal_rag":
        if project_root.parent == project_root:
            raise RuntimeError("Could not find 'multimodal_rag' in parent directories.")
        project_root = project_root.parent

    for turn in conversation_history:
        # Add user's question
        doc.add_paragraph(f"🧑 You: {turn['question']}", style="Intense Quote")

        answer_text = turn['answer']
        # Find all images in markdown format ![alt](path)
        img_matches = re.findall(r"!\[.*?\]\((.*?)\)", answer_text)

        if img_matches:
            # Split text around images
            parts = re.split(r"!\[.*?\]\(.*?\)", answer_text)
            for i, part in enumerate(parts):
                if part.strip():
                    doc.add_paragraph(f"🤖 Allion: {part.strip()}")

                if i < len(img_matches):
                    rel_path = Path(img_matches[i])
                    # 🔹 Preserve duplicated 'output/markdowns/output/markdowns'
                    fixed_path = project_root / "output" / "markdowns" / rel_path
                    img_path = fixed_path.resolve()

                    if img_path.exists():
                        doc.add_picture(str(img_path), width=Inches(5))
                    else:
                        doc.add_paragraph(f"[Image not found: {img_path}]")
        else:
            # No images, just add the text
            doc.add_paragraph(f"🤖 Allion: {answer_text}")

    # Save the Word document
    doc.save(output_path)
    print(f"✅ Conversation saved to {output_path}")




# -------------------------
# Interactive session
# -------------------------
def main():
    print("🚗 Allion Automotive Assistant Started! Type 'exit' to quit.\n")
    assistant = Assistant()

    while True:
        question = input("🧑 You: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("👋 Ending session.")
            # Save final conversation before exit
            save_conversation_to_word(assistant.memory["conversation_history"])
            break

        # Ask assistant
        answer = assistant.ask(question)

        # Display answer
        print(f"\n🤖 Allion: {answer['answer']}\n")

        # Show source documents if available
        if answer.get("source_documents"):
            print("📄 Source Documents:")
            for doc in answer["source_documents"]:
                print(f" - Page: {doc['page_number']}")

        # Save conversation after every turn
        save_conversation_to_word(assistant.memory["conversation_history"])

if __name__ == "__main__":
    main()