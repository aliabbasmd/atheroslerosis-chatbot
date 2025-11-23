import json
import os
import logging
from hipporag import HippoRAG

# --- Configuration ---
CORPUS_PATH = "./data/athero_corpus.json"
SAVE_DIR = './storage/athero_kg'
LLM_MODEL_NAME = 'phi3:mini' 
LLM_BASE_URL = "http://localhost:11434/v1" 
EMBEDDING_MODEL_NAME = 'GritLM/GritLM-7B' 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    """Loads the corpus data (list of dicts) from the JSON file."""
    if not os.path.exists(path):
        logging.error(f"Corpus file not found at {path}. Please run preprocess.py first.")
        return []
    with open(path, 'r') as f:
        return json.load(f)

def run_hipporag_indexing():
    """Initializes HippoRAG and runs the indexing process."""
    
    # Load your documents (list of dicts: [{'text': '...', 'title': '...'}])
    docs_dicts = load_data(CORPUS_PATH)
    if not docs_dicts:
        logging.error("Indexing aborted due to missing data.")
        return

    logging.info(f"Loaded {len(docs_dicts)} passages for indexing.")
    
    # -----------------------------------------------------------------
    # FIX: Convert the list of dictionaries into the list of raw strings 
    # that the hipporag.index() function expects, avoiding the TypeError.
    docs_raw_strings = [d['text'] for d in docs_dicts]
    logging.info(f"Converted docs to list of {len(docs_raw_strings)} raw text strings.")
    # -----------------------------------------------------------------
    
    # 1. Startup a HippoRAG instance (using only accepted arguments)
    hipporag = HippoRAG(
        save_dir=SAVE_DIR,
        llm_model_name=LLM_MODEL_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        llm_base_url=LLM_BASE_URL,
        #timeout=180 # Generous timeout for local inference
    )

    logging.info(f"Starting Knowledge Graph indexing with {LLM_MODEL_NAME} via {LLM_BASE_URL}")

    # 2. Run indexing (builds and saves the KG)
    hipporag.index(docs=docs_raw_strings)

    logging.info(f"Indexing complete. Knowledge Graph saved to {SAVE_DIR}")

if __name__ == "__main__":
    run_hipporag_indexing()
