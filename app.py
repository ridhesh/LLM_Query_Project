import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
import tempfile
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

# Set up logging for better traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Google API for Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    """
    Extracts text from uploaded PDF files and returns the concatenated text.
    """
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    logger.warning(f"Page {pdf_reader.pages.index(page)} in {pdf.name} has no extractable text.")
        except Exception as e:
            logger.error(f"Error processing {pdf.name}: {e}")
    return text


def get_text_chunks(text):
    """
    Splits the text into smaller chunks using the RecursiveCharacterTextSplitter.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}")
        return []


def get_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks and saves it locally.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        logger.info("Vector store created and saved locally.")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")


def get_conversational_chain():
    """
    Returns a conversational chain to handle question answering based on provided context.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in the provided context, just say, "Answer is not available in the context," don't provide the wrong answer.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    """
    Handles user input, performs similarity search on the vector store,
    and gets the response from the conversational chain.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        if docs:
            chain = get_conversational_chain()
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            logger.info(f"Response generated for question: {user_question}")
            st.write("Reply: ", response["output_text"])
        else:
            st.write("No relevant information found in the context.")
            logger.warning(f"No relevant documents found for question: {user_question}")
    except Exception as e:
        st.write("An error occurred while processing your request.")
        logger.error(f"Error handling user input: {e}")


async def process_pdf_files(pdf_docs):
    """
    Asynchronously process PDFs to extract text, split into chunks, and generate vector store.
    """
    raw_text = get_pdf_text(pdf_docs)
    if raw_text.strip():
        text_chunks = get_text_chunks(raw_text)
        if text_chunks:
            get_vector_store(text_chunks)
            return True
        else:
            return False
    else:
        return False


def main():
    """
    Main Streamlit app function that initializes the page and handles interactions.
    """
    try:
        st.set_page_config(page_title="Chat PDF", page_icon=":books:")
        st.header("Ask Questions from the Given PDF Data")

        user_question = st.text_input("Ask a question based on the uploaded PDFs:")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            
            if pdf_docs:
                if st.button("Submit & Process"):
                    with st.spinner("Processing..."):
                        # Async processing to avoid blocking the UI
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(process_pdf_files(pdf_docs))
                        if result:
                            st.success("Processing complete! You can now ask questions.")
                        else:
                            st.error("Failed to process the PDFs or extract text.")
            else:
                st.warning("Please upload PDF files to process.")

    except Exception as e:
        st.error("An error occurred while initializing the app.")
        logger.error(f"Error in main function: {e}")
if __name__ == "__main__":
    main()