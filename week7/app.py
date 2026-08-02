"""
app.py
Streamlit interface for the RAG Document Question Answering system.
Upload a PDF, wait for it to be indexed, then ask questions grounded
in the document's content.
"""

import streamlit as st
from document_processor import process_document
from vectorstore import add_chunks, clear_collection
from chatbot import answer_question

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG Document Question Answering")
st.caption("Upload a PDF or text file, then ask questions about its content. Runs fully local via Ollama + ChromaDB.")

if "indexed" not in st.session_state:
    st.session_state.indexed = False

with st.sidebar:
    st.header("1. Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.button("Process document"):
            with st.spinner("Extracting text and building embeddings..."):
                clear_collection()  # start fresh for each new document
                chunks = process_document(uploaded_file, uploaded_file.name)
                num_chunks = add_chunks(chunks, source_name=uploaded_file.name)
                st.session_state.indexed = True
            st.success(f"Indexed {num_chunks} chunks from {uploaded_file.name}")

st.header("2. Ask a question")

if not st.session_state.indexed:
    st.info("Upload and process a PDF from the sidebar to get started.")
else:
    question = st.text_input("Your question")
    if question:
        with st.spinner("Retrieving relevant context and generating an answer..."):
            result = answer_question(question)

        st.subheader("Answer")
        st.write(result["answer"])

        if result["sources"]:
            with st.expander("View source chunks used"):
                for src in result["sources"]:
                    st.markdown(f"**{src['source']} — chunk {src['chunk_index']}**")
                    st.write(src["text"])
                    st.divider()
