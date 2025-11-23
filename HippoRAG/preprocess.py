import json
import os
import re
from pypdf import PdfReader
from glob import glob

# --- Configuration ---
# The base path to your directory of PDFs (must be the same as your symbolic link)
PDF_DIR_PATH = "~/athero_data" 
# The output file name 
OUTPUT_FILE = "data/athero_corpus.json"
# Chunk size (words per passage)
CHUNK_SIZE = 500 

def get_pdf_text_chunks(abs_path, chunk_size=CHUNK_SIZE):
    """Extracts text from a single PDF and splits it into chunks."""
    try:
        reader = PdfReader(abs_path)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        
        # Simple cleaning: normalize whitespace
        full_text = re.sub(r'\s+', ' ', full_text).strip()

        # Segment into chunks of size CHUNK_SIZE
        words = full_text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    except Exception as e:
        print(f"Error processing PDF {abs_path}: {e}")
        return []

def create_corpus_json(pdf_dir_path, output_path):
    """Iterates through all PDFs in a directory and creates the JSON corpus file."""
    abs_pdf_dir = os.path.expanduser(pdf_dir_path)
    
    if not os.path.exists(abs_pdf_dir):
        print(f"Error: PDF directory not found at {abs_pdf_dir}")
        return

    # Find all PDF files in the directory
    pdf_files = glob(os.path.join(abs_pdf_dir, "*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in {abs_pdf_dir}")
        return

    corpus = []
    chunk_idx = 0
    
    print(f"Found {len(pdf_files)} PDFs. Starting extraction...")

    # Process each PDF file
    for file_path in pdf_files:
        # Use the filename (without extension) as the document title
        doc_title = os.path.splitext(os.path.basename(file_path))[0]
        
        chunks = get_pdf_text_chunks(file_path)
        
        for chunk in chunks:
            corpus.append({
                "title": doc_title,
                "text": chunk,
                "idx": chunk_idx
            })
            chunk_idx += 1
        
        print(f"Processed '{doc_title}' into {len(chunks)} chunks.")

    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(corpus, f, indent=4)
    
    print(f"\nSuccessfully created total corpus with {len(corpus)} chunks at {output_path}")

create_corpus_json(PDF_DIR_PATH, OUTPUT_FILE)
