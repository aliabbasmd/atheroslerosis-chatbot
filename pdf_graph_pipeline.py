import streamlit as st
import os
import json
import networkx as nx
import gc
from pypdf import PdfReader
from streamlit_agraph import agraph, Node, Edge, Config

# --- DISABLE TELEMETRY ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# --- IMPORTS ---
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
SOURCE_PDF_FOLDER = "/home/abbas/athero_data"
INTERMEDIATE_FACTS_FOLDER = "/mnt/d/rag_data/extracted_facts"
DB_PATH = "/mnt/d/rag_data/chroma_graph_db"
MODEL_NAME = "phi3:mini" 

st.set_page_config(page_title="PDF to HippoRAG", layout="wide")
st.title("📄➡️🕸️ PDF to HippoRAG (Final)")

# --- SETUP ---
if not os.path.exists(INTERMEDIATE_FACTS_FOLDER):
    os.makedirs(INTERMEDIATE_FACTS_FOLDER)

# --- HELPER: BUILD GRAPH FROM DISK ---
@st.cache_resource
def load_combined_graph_from_disk(folder_path):
    G = nx.DiGraph()
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    for f in files:
        try:
            with open(os.path.join(folder_path, f), 'r') as file:
                data = json.load(file)
                if 'docs' in data:
                    for doc in data['docs']:
                        if 'extracted_triples' in doc:
                            for t in doc['extracted_triples']:
                                if isinstance(t, dict):
                                    s, r, o = t.get('subject'), t.get('relation'), t.get('object')
                                elif isinstance(t, list) and len(t) >= 3:
                                    s, r, o = t[0], t[1], t[2]
                                else: continue
                                
                                if s and o:
                                    s_clean = str(s).strip()[:30]
                                    o_clean = str(o).strip()[:30]
                                    r_clean = str(r).strip()[:20]
                                    G.add_edge(s_clean, o_clean, relation=r_clean)
        except: pass
    return G

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["1. Extract Facts", "2. Chat with Graph", "3. Smart Visualization"])

# --- TAB 1: EXTRACTION ---
with tab1:
    st.subheader("Step 1: Extract Knowledge")
    def extract_text(pdf_path):
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages: text += page.extract_text() + "\n"
            return text
        except: return None

    if os.path.exists(SOURCE_PDF_FOLDER):
        pdf_files = [f for f in os.listdir(SOURCE_PDF_FOLDER) if f.lower().endswith('.pdf')]
        existing_jsons = [f for f in os.listdir(INTERMEDIATE_FACTS_FOLDER) if f.endswith('.json')]
        st.info(f"Found {len(pdf_files)} PDFs. {len(existing_jsons)} already processed.")
        to_process = [f for f in pdf_files if f.replace('.pdf','.json') not in existing_jsons]
        
        if to_process and st.button("Start Extraction"):
            bar = st.progress(0)
            llm = ChatOllama(model=MODEL_NAME, format="json", temperature=0)
            splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
            for i, pdf in enumerate(to_process):
                text = extract_text(os.path.join(SOURCE_PDF_FOLDER, pdf))
                if text:
                    chunks = splitter.split_text(text)[:20] 
                    file_triples = []
                    for chunk in chunks:
                        prompt = f"Extract medical facts as JSON list: [{{'subject':'X', 'relation':'Y', 'object':'Z'}}]. Text: {chunk}"
                        try:
                            res = llm.invoke(prompt).content
                            start, end = res.find('['), res.rfind(']')+1
                            if start != -1: file_triples.extend(json.loads(res[start:end]))
                        except: pass
                    with open(os.path.join(INTERMEDIATE_FACTS_FOLDER, pdf.replace('.pdf','.json')), 'w') as f:
                        json.dump({"docs": [{"extracted_triples": file_triples}], "source": pdf}, f)
                bar.progress((i+1)/len(to_process))
                gc.collect()
            st.success("Extraction Done!")
            st.rerun()

# --- TAB 2: CHAT ---
with tab2:
    st.subheader("Step 2: Query the Graph")
    with st.spinner("Loading Graph Matrix..."):
        G = load_combined_graph_from_disk(INTERMEDIATE_FACTS_FOLDER)
        st.caption(f"Active Graph: {len(G.nodes())} Nodes")
    
    embed = OllamaEmbeddings(model=MODEL_NAME)
    vector_store = Chroma(
        embedding_function=embed,
        collection_name="athero_combined_graph",
        persist_directory=DB_PATH
    )
    
    if st.button("Re-Index Graph"):
        with st.spinner("Indexing nodes..."):
            nodes = list(G.nodes())
            docs = [Document(page_content=n, metadata={"node": n}) for n in nodes]
            for i in range(0, len(docs), 100):
                vector_store.add_documents(docs[i:i+100])
            st.success("Indexed!")
    
    query = st.chat_input("Ask about the research...")
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: 
        with st.chat_message(m["role"]): st.markdown(m["content"])
    
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
                    results = retriever.invoke(query)
                    start_nodes = [d.metadata['node'] for d in results]
                    facts = []
                    for node in start_nodes:
                        if node in G:
                            for nbr, attr in G[node].items(): facts.append(f"{node} -> {attr.get('relation')} -> {nbr}")
                            for u, v, d in G.in_edges(node, data=True): facts.append(f"{u} -> {d.get('relation')} -> {node}")
                    unique_facts = list(set(facts))[:50]
                    context_str = "\n".join(unique_facts)
                    llm = ChatOllama(model=MODEL_NAME)
                    final_prompt = f"Facts:\n{context_str}\n\nQuestion: {query}"
                    res = llm.invoke(final_prompt).content
                    st.markdown(res)
                    st.session_state.messages.append({"role": "assistant", "content": res})
                except Exception as e: st.error(f"Error: {e}")


# --- TAB 3: SMART VISUALIZATION ---
with tab3:
    st.subheader("3D Knowledge Graph Explorer")
    
    # Load Graph
    G_viz = load_combined_graph_from_disk(INTERMEDIATE_FACTS_FOLDER)
    
    # Sidebar Controls
    with st.container():
        c1, c2, c3 = st.columns([2, 1, 1])
        focus_mode = c1.radio("Focus Mode:", ["Topic Search", "Most Influential Nodes (PageRank)"])
        max_nodes = c2.slider("Max Nodes", 10, 100, 30)
    
    sub_G = None
    viz_caption = "Graph Visualization"
    center_node = "" # Initialize variable to prevent NameError
    
    # 1. PageRank Mode
    if focus_mode == "Most Influential Nodes (PageRank)":
        if st.button("Calculate Influence"):
            with st.spinner("Ranking nodes..."):
                try:
                    pagerank = nx.pagerank(G_viz)
                    top_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:max_nodes]
                    sub_G = G_viz.subgraph(top_nodes)
                    viz_caption = f"Top {max_nodes} Most Influential Nodes"
                    st.success(f"Showing top {max_nodes} central nodes.")
                except Exception as e:
                    st.error(f"PageRank failed (needs scipy): {e}")

    # 2. Topic Search Mode
    else:
        search_term = st.text_input("Enter Topic (e.g., Atherosclerosis):")
        if st.button("Zoom to Topic") and search_term:
            matches = [n for n in G_viz.nodes() if search_term.lower() in str(n).lower()]
            if matches:
                subset = set(matches)
                for m in matches:
                    subset.update(list(G_viz.neighbors(m)))
                final_subset = list(subset)[:max_nodes]
                sub_G = G_viz.subgraph(final_subset)
                viz_caption = f"Nodes related to '{search_term}'"
                center_node = search_term # Set for caption
                st.success(f"Found {len(matches)} matches.")
            else:
                st.warning("No nodes found matching that term.")

    # Render
    if sub_G:
        nodes = []
        edges = []
        for n in sub_G.nodes():
            degree = sub_G.degree(n)
            size = 15 + (degree * 3)
            color = "#FF4B4B" if degree > 5 else "#0083B8"
            nodes.append(Node(id=n, label=n, size=size, color=color))
        
        for s, t in sub_G.edges():
            r = sub_G[s][t].get('relation', '')
            # FIX: Explicitly set type here to force rendering
            edges.append(Edge(source=s, target=t, label=r, type="CURVE_SMOOTH", strokeWidth=4, color="#333333"))

        # CONFIG: High Contrast Mode
        config = Config(
            width=900,
            height=600,
            directed=True, 
            physics=True, 
            hierarchy=False,
            nodeHighlightBehavior=True, 
            highlightColor="#F7A7A6",
            # Global Link Settings
            link={
                "renderLabel": True,
                "labelProperty": "label",
                "strokeWidth": 8,       
                "color": "#333333",     
                "type": "CURVE_SMOOTH"
            }
        )
        
        st.caption(viz_caption)
        agraph(nodes=nodes, edges=edges, config=config)
