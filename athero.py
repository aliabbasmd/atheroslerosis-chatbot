import streamlit as st
import os
import zipfile
import logging
from pathlib import Path

# Imports for HippoRAG and the Google connector
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import ServiceContext, StorageContext, Settings
from llama_index.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Note: You must commit the hipporag source folder or install it in requirements.txt
# import hipporag 

# --- Configuration & Setup ---

# Use a specific, supported local embedding model name
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

# Define file paths relative to the Streamlit app root
KG_ZIP_PATH = "gemini_kg_index.zip"
KG_EXTRACT_DIR = Path("./kg_data")
# The final directory containing the LlamaIndex persistence files
FINAL_KG_PATH = KG_EXTRACT_DIR / 'HippoRAG/storage/gemini_kg' 

# Set up logging (Streamlit is often noisy without this)
logging.basicConfig(level=logging.INFO)

# --- Knowledge Graph Loading ---

@st.cache_resource
def setup_rag_engine():
    # 1. Check for API Key (from Streamlit Secrets)
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("GEMINI_API_KEY not found in Streamlit Secrets. Cannot connect to LLM.")
        return None
    
    # Set the environment variable for the underlying LlamaIndex/HippoRAG compatibility
    os.environ["OPENAI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    
    # 2. Unzip the KG data if not already extracted
    if not FINAL_KG_PATH.exists():
        try:
            with zipfile.ZipFile(KG_ZIP_PATH, 'r') as zip_ref:
                # Extracts files into the ./kg_data folder
                zip_ref.extractall(KG_EXTRACT_DIR) 
            st.success("Knowledge Graph files extracted successfully.")
        except Exception as e:
            st.error(f"Error extracting KG files. Did you commit {KG_ZIP_PATH}? Error: {e}")
            return None

    # 3. Initialize LLM and Embedding Model
    # Note: Using HuggingFaceEmbedding requires the model to be downloaded or available.
    try:
        llm = GoogleGenAI(model=LLM_MODEL)
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    except Exception as e:
        st.error(f"Error initializing models. Check requirements.txt. Error: {e}")
        return None

    # 4. Configure LlamaIndex Service Context and Load KG Index
    Settings.llm = llm
    Settings.embed_model = embed_model
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(FINAL_KG_PATH))
        # Load the Knowledge Graph from the saved directory
        kg_index = KnowledgeGraphIndex.from_client_and_storage(
            storage_context=storage_context,
            service_context=service_context
        )
        return kg_index.as_query_engine(
            retriever_mode="keyword", # Uses graph traversal
            response_mode="tree_summarize",
            verbose=False
        )
    except Exception as e:
        st.error(f"Error loading Knowledge Graph index. {e}")
        return None

# --- Streamlit Chatbot Interface ---

st.title("🩺 Atherosclerosis HippoRAG Expert")

# Setup the query engine (runs only once thanks to @st.cache_resource)
query_engine = setup_rag_engine()

if query_engine is None:
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the HippoRAG expert on your atherosclerosis documents. Ask me a complex question about plaque, inflammation, or exercise!"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and RAG query
if prompt := st.chat_input("Enter your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Knowledge Graph..."):
            try:
                # Execute the RAG query against the HippoRAG KG
                response = query_engine.query(prompt)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
            except Exception as e:
                error_msg = f"An error occurred during query execution. Error: {e}"
                st.warning(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
