Atherosclerosis Research AI Station

A local, privacy-first AI toolkit for analyzing medical research papers (PDFs) and structured data (CSVs) using RAG (Retrieval-Augmented Generation) and Knowledge Graphs.

📂 Project Structure & File Guide

Here is what each Python script does:

1. Core Applications

    rag_app.py

        Purpose: The standard "Chat with your PDF" tool.

        Best for: Uploading a specific PDF or CSV and asking general questions like "What does this paper say about lipids?".

        Tech: Uses Vector RAG (ChromaDB + Phi-3/Llama-3).

    data_analyst.py

        Purpose: A "Text-to-SQL" style tool for structured data.

        Best for: Analyzing Excel/CSV files (e.g., Patient demographics).

        Capabilities: Can filter, count, sort, and clean data (e.g., "Find all female patients under 30 with spiritual traits"). It writes and runs its own Python code.

    graph_rag.py

        Purpose: An advanced "Knowledge Graph" explorer.

        Best for: Querying pre-extracted JSON knowledge graphs (like the ones from GPT-4o or Gemini).

        Features: Multi-hop reasoning (Node A -> Node B -> Node C).

2. Advanced Pipelines

    pdf_graph_pipeline.py (The Master Script)

        Purpose: The complete "End-to-End" factory.

        Workflow:

            Reads raw PDFs from your athero_data folder.

            Extracts scientific facts ("Triples") using local AI.

            Saves extraction to D: drive (Failsafe).

            Builds a searchable Graph Database.

            Visualizes the network in 3D with centrality ranking.

3. Utility Tools

    inspect_json.py

        Purpose: A diagnostic tool to "X-Ray" your JSON data files.

        Use it when: You want to see the Top 10 topics in a file or check if a file is empty/corrupt.

    pdf_converter_v2.py

        Purpose: A specialized scraper for "Registration Form" style PDFs.

        Use it when: You have messy forms that need to be converted into a clean CSV for the data_analyst.py tool.

🚀 Quick Start

1. Activate Environment
Bash

conda activate rag_env

2. Run an App
Bash

streamlit run pdf_graph_pipeline.py

3. Typical Workflow

    Step 1: Put new papers in ~/athero_data.

    Step 2: Run pdf_graph_pipeline.py to extract facts and build the graph.

    Step 3: Use the Visualize tab to explore connections.

    Step 4: Use rag_app.py if you just need to read one specific paper quickly.

🛠️ Configuration

    Source Data: ~/athero_data

    Graph Database: /mnt/d/rag_data/chroma_graph_db

    Doc Database: /mnt/d/rag_data/chroma_doc_db

    Extracted JSONs: /mnt/d/rag_data/extracted_facts

📦 Requirements

    System: WSL2 (Ubuntu) with NVIDIA GPU Support.

    Model: Ollama running phi3:mini or llama3.

    Python: 3.11+
