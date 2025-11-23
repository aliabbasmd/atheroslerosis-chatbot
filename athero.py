import streamlit as st
import os
import zipfile
import logging
from pathlib import Path

# --- Final, Clean Import Structure (Avoids Ambiguity) ---
from llama_index.core.settings import Settings 
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.service_context import ServiceContext

# These imports are correct and should be kept simple:
from llama_index.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# ---------------------------------------------------------------------------------

# Configuration
# NOTE: The name of the zip file must match the file you pushed to GitHub.
KG_ZIP_PATH = "gemini_kg_project.zip" 
KG_EXTRACT_DIR = Path("./kg_data")
# The final directory containing the LlamaIndex persistence files
FINAL_KG_PATH = KG_EXTRACT_DIR / 'HippoRAG/storage/gemini_kg' 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Knowledge Graph Loading ---

@st.cache_resource
def setup_rag_engine():
    # 1. Check for API Key (from Streamlit Secrets)
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("GEMINI_API_KEY not found in Streamlit Secrets. Cannot connect to LLM.")
        return None
    
    # Configure authentication for underlying libraries
    os.environ["OPENAI_API_KEY"] = st.secrets["GEMINI_API_KEY"] 
    
    # 2. Unzip the KG data if not already extracted (CRUCIAL STEP)
    if not FINAL_KG_PATH.exists():
        st.info(f"Extracting {KG_ZIP_PATH}...")
        try:
            with zipfile.ZipFile(KG_ZIP_PATH, 'r') as zip_ref:
                # Extracts files into the ./kg_data folder
                zip_ref.extractall(KG_EXTRACT_DIR) 
            st.success("Knowledge Graph files extracted.")
        except Exception as e:
            st.error(f"Error extracting KG files. Ensure '{KG_ZIP_PATH}' is in the repo root. Error: {e}")
            return None

    # 3. Initialize stable models
    try:
        llm = GoogleGenAI(model="gemini-2.5-flash")
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error initializing models. Check requirements.txt. Error: {e}")
        return None

    # 4. Configure LlamaIndex Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    # 5. Load the Knowledge Graph Index
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(FINAL_KG_PATH))
        
        # Load the index using standard LlamaIndex KG logic
        kg_index = KnowledgeGraphIndex.from_client_and_storage(
            storage_context=storage_context,
            service_context=service_context # Re-added service_context for robustness
        )
        # Return the configured query engine
        return kg_index.as_query_engine(
            retriever_mode="keyword", 
            response_mode="tree_summarize",
            verbose=False
        )
    except Exception as e:
        st.error(f"Error loading Knowledge Graph index. Ensure data integrity. Error: {e}")
        return None

# --- Streamlit Chatbot Interface ---

st.title("🩺 Atherosclerosis RAG Chatbot (Powered by Gemini)")

# Setup the query engine (runs only once thanks to @st.cache_resource)
query_engine = setup_rag_engine()

if query_engine is None:
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the RAG expert on your atherosclerosis documents. Ask me a question!"}
    ]

# Display chat messages from history on app rerun
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
                # Execute the RAG query against the Knowledge Graph
                response = query_engine.query(prompt)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
            except Exception as e:
                error_msg = f"An error occurred during query execution. Error: {e}"
                st.warning(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
