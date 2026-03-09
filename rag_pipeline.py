import os
os.environ["USE_TF"] = "0"

import streamlit as st
import uuid
import tempfile
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.set_page_config(page_title="RAG PDF Q&A", page_icon="📄")
st.title("📄 RAG PDF Question Answering App")
st.write("Upload any PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # create a fresh temp file every time
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_pdf_path = tmp_file.name

    # load the uploaded PDF
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    # split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # create a fresh Chroma collection every upload
    collection_name = f"pdf_{uuid.uuid4().hex}"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )

    st.success(f"Processed: {uploaded_file.name}")

    query = st.text_input("Ask a question about the PDF")

    if query:
        results = vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in results])

        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}

If the answer is not in the context, say:
"The document does not contain the answer."
"""

        response = model.generate_content(prompt)

        st.subheader("Answer")
        st.write(response.text)

        with st.expander("See retrieved chunks"):
            for i, doc in enumerate(results, 1):
                st.write(f"**Chunk {i}:**")
                st.write(doc.page_content)
                st.write("---")

    # optional cleanup
    try:
        os.remove(temp_pdf_path)
    except Exception:
        pass
