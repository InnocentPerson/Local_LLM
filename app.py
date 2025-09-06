# app.py
import os
import streamlit as st
import torch

# ✅ Updated imports to avoid deprecation warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from PyPDF2 import PdfReader
import docx2txt

# ✅ Set Chroma DB path on D: to save C: space
DB_PATH = "D:/OLLAMA_MAIN/db"

# ✅ Force CPU if CUDA not available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"📌 Running on: **{DEVICE.upper()}**")

# Step 1. File uploader
uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    # Step 2. Read file
    if file_type == "txt":
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    elif file_type == "pdf":
        pdf = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file_type == "docx":
        text = docx2txt.process(uploaded_file)
    else:
        st.error("Unsupported file type!")
        st.stop()

    # Step 3. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Step 4. Create Chroma DB (persistent on D:)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}  # ✅ Force CPU/GPU here
    )
    db = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=DB_PATH)

    # Step 5. Build QA system
    llm = Ollama(model="mistral")
    qa = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

    # Step 6. Ask questions
    st.title("📚 Chat with My File")
    user_q = st.text_input("Ask a question:")
    if st.button("Submit") and user_q:
        with st.spinner("Thinking..."):
            answer = qa.run(user_q)
        st.success(answer)
