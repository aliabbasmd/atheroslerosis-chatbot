import os
import logging
import json
import re
from glob import glob
from pypdf import PdfReader

# Imports for HippoRAG and the Google connector
from llama_index.llms.google_genai import GoogleGenAI
from hipporag import HippoRAG

# --- Configuration ---
# IMPORTANT: Ensure both OPENAI_API_KEY and GEMINI_API_KEY are set in Bash!
LOCAL_PDF_DIR = "/mnt/c/Users/abbas/Downloads/athero"  # Your definitive data path
CORPUS_PATH = "./data/gemini_corpus.json"
SAVE_DIR = './storage/gemini_kg' 

LLM_MODEL = "gemini-2.5-flash" 
# Use the model name that the HippoRAG code will recognize (GritLM is commonly supported)
EMBED_MODEL = "GritLM/GritLM-7B" 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_corpus():
    """Extracts text from all PDFs and creates the JSON corpus."""
    if not os.path.exists(LOCAL_PDF_DIR):
        logging.error(f"Local PDF directory not found: {LOCAL_PDF_DIR}. Cannot create corpus.")
        return []

    pdf_files = glob(os.path.join(LOCAL_PDF_DIR, "*.pdf"))
    if not pdf_files:
        logging.error(f"No PDF files found in {LOCAL_PDF_DIR}. Cannot create corpus.")
        return []

    corpus = []
    chunk_idx = 0
    chunk_size = 500

    for file_path in pdf_files:
        try:
            reader = PdfReader(file_path)
            full_text = "".join(page.extract_text() or "" for page in reader.pages)
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            words = full_text.split()
            doc_title = os.path.splitext(os.path.basename(file_path))[0]

            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                corpus.append({"title": doc_title, "text": chunk, "idx": chunk_idx})
                chunk_idx += 1
        except Exception as e:
            logging.warning(f"Failed to process {file_path}: {e}")
            continue

    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    with open(CORPUS_PATH, 'w') as f:
        json.dump(corpus, f, indent=4)
    
    logging.info(f"Created corpus with {len(corpus)} total chunks.")
    return [d['text'] for d in corpus]


def run_gemini_indexing():
    """Runs the HippoRAG indexing using the Gemini API."""
    
    # Check for the key name that the underlying framework demands
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY environment variable (containing Gemini Key) not set. Indexing aborted.")
        return

    # 1. Prepare data (returns list of raw text strings)
    docs_raw_strings = create_corpus()
    if not docs_raw_strings:
        return

    # 2. Initialize HippoRAG with Gemini settings
    hipporag = HippoRAG(
        save_dir=SAVE_DIR,
        llm_model_name=LLM_MODEL,
        embedding_model_name=EMBED_MODEL,
        # Only use accepted arguments
    )

    logging.info(f"Starting Knowledge Graph indexing with {LLM_MODEL} via Gemini API.")

    # 3. Run indexing
    hipporag.index(docs=docs_raw_strings)

    logging.info(f"Indexing complete. Knowledge Graph saved to {SAVE_DIR}")
    
if __name__ == "__main__":
    run_gemini_indexing()
