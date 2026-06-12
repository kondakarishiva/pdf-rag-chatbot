import streamlit as st
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)

st.title("📚 PDF RAG Chatbot")

api_key = st.text_input(
    "Enter Gemini API Key",
    type="password"
)

if api_key:

    try:

        os.environ["GOOGLE_API_KEY"] = api_key

        loader = TextLoader("document.txt")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(
            chunks,
            embeddings
        )

        retriever = db.as_retriever()

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever
        )

        st.success("✅ RAG System Ready")

        query = st.text_input("Ask a Question")

        if query:

            with st.spinner("Thinking..."):

                result = qa.invoke(
                    {"query": query}
                )

                st.subheader("Answer")

                st.write(
                    result["result"]
                )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )
