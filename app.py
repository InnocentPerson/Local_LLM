# app.py
import os
import uuid
import streamlit as st
import torch

# PDF / DOCX reading
from pypdf import PdfReader
import docx2txt

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# 🔥 NEW (for conversational memory)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# -----------------------------------------------------------
# ✅ SETTINGS (PORTABLE)
# -----------------------------------------------------------

DB_BASE_PATH = "./db"
os.makedirs(DB_BASE_PATH, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"📌 Running on: **{DEVICE.upper()}**")


# -----------------------------------------------------------
# 🔥 HELPER FUNCTIONS
# -----------------------------------------------------------

def load_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "txt":
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    elif file_type == "pdf":
        pdf = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages)

    elif file_type == "docx":
        text = docx2txt.process(uploaded_file)

    else:
        st.error("Unsupported file type!")
        st.stop()

    if not text.strip():
        st.error("❌ No readable text found (scanned PDF maybe).")
        st.stop()

    return text


def split_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = [c for c in splitter.split_text(text) if c.strip()]

    if not chunks:
        st.error("❌ Could not create text chunks.")
        st.stop()

    return chunks


def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # stable on all PCs
    )

    unique_db_path = os.path.join(DB_BASE_PATH, f"session_{uuid.uuid4()}")
    os.makedirs(unique_db_path, exist_ok=True)

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=unique_db_path
    )

    return db


def build_qa_chain(db):
    llm = Ollama(model="mistral", temperature=0.3)

    # 🧠 Conversation memory (OPTION 1)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory
    )

    return qa


# -----------------------------------------------------------
# 🚀 STREAMLIT UI
# -----------------------------------------------------------

st.title("📚 Chat With Your File (Local LLM + ChromaDB)")
st.write("Upload a document and chat with it like ChatGPT.")

uploaded_file = st.file_uploader(
    "Upload PDF / TXT / DOCX",
    type=["txt", "pdf", "docx"]
)

# ---------------- SESSION STATE ----------------

if "qa" not in st.session_state:
    st.session_state.qa = None

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------- DOCUMENT PROCESSING ----------------

if uploaded_file and st.session_state.qa is None:
    with st.spinner("📄 Reading document..."):
        text = load_text_from_file(uploaded_file)

    chunks = split_into_chunks(text)
    st.success(f"📌 Document split into {len(chunks)} chunks")

    with st.spinner("🔧 Creating vector database..."):
        db = create_vector_db(chunks)

    with st.spinner("🤖 Initializing AI model with memory..."):
        st.session_state.qa = build_qa_chain(db)


# ---------------- CHAT UI (ChatGPT STYLE) ----------------

# ---------------- CHAT UI (ChatGPT STYLE) ----------------

if st.session_state.qa:

    # 🔄 Clear Chat Button
    if st.button("🧹 Clear Chat / New Chat"):
        st.session_state.history = []

        # Clear LLM conversational memory
        if hasattr(st.session_state.qa, "memory"):
            st.session_state.qa.memory.clear()

        st.success("Chat cleared! You can start a new conversation.")

    user_q = st.chat_input("Ask something about your document...")

    if user_q:
        with st.spinner("Thinking..."):
            result = st.session_state.qa({"question": user_q})
            answer = result["answer"]

        st.session_state.history.append(("user", user_q))
        st.session_state.history.append(("assistant", answer))

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

else:
    st.info("📄 Upload a file to begin.")

