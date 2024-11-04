import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Streamlit app title
st.title("PDF Q&A with LangChain and FAISS")

# PDF File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # loading the LLM
    llm = Ollama(
        model="llama3",
        temperature=0
    )

    # loading the document
    loader = PyPDFLoader("uploaded_pdf.pdf")
    documents = loader.load()

    # create document chunks
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size=1000,
                                          chunk_overlap=200)

    text_chunks = text_splitter.split_documents(documents)

    # loading the vector embedding model
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)

    # retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever())

    # Input box for the question
    question = st.text_input("Ask a question about the document")

    if question:
        # Process the question through the QA chain
        response = qa_chain.invoke({"query": question})
        # Display the result
        st.write("### Answer:")
        st.write(response["result"])
