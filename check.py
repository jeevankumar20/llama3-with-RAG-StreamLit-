import streamlit as st
from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

st.title("PDF Q&A with LangChain and FAISS")

# PDF File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    st.write("File uploaded successfully!")

    # Save the uploaded file temporarily
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("PDF saved locally.")

    # Load the PDF document
    try:
        loader = PyPDFLoader("uploaded_pdf.pdf")
        documents = loader.load()
        st.write(f"Loaded {len(documents)} pages from the PDF.")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        st.stop()

    # Create document chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)
    st.write(f"Document split into {len(text_chunks)} chunks.")

    # Initialize embeddings and knowledge base
    try:
        embeddings = HuggingFaceEmbeddings()
        knowledge_base = FAISS.from_documents(text_chunks, embeddings)
        st.write("Knowledge base created successfully.")
    except Exception as e:
        st.error(f"Error creating knowledge base: {e}")
        st.stop()

    # Load the language model
    try:
        llm = Ollama(model="llama3", temperature=0)
        st.write("Language model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading language model: {e}")
        st.stop()

    # Set up the retrieval QA chain
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=knowledge_base.as_retriever()
        )
        st.write("QA chain created successfully.")
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        st.stop()

    # Input box for questions
    question = st.text_input("Ask a question about the document")
    if question:
        # Process the question through the QA chain
        try:
            response = qa_chain.invoke({"query": question})
            st.write("### Answer:")
            st.write(response["result"])
        except Exception as e:
            st.error(f"Error processing question: {e}")
