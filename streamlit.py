import streamlit as st
st.set_page_config(page_title="PDF Chat AI", page_icon=":robot_face:")

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Firebase authentication
import firebase_admin
from firebase_admin import auth
from firebase_admin import credentials

# Load environment variables
load_dotenv()

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        st.success("Welcome To PDF VERSE AI BY SVECTOR")
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, api_key):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
def user_input(user_question, api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        print(response)
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error during user input handling: {e}")


def main():
    # Initialize session state for user authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        # Firebase authentication
        st.title("Sign In / Sign Up")
        choice = st.radio("Select an option", ["Sign In", "Sign Up"])

        if choice == "Sign In":
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Sign In"):
                try:
                    user = auth.get_user_by_email(email)
                    token = auth.create_custom_token(user.uid)
                    st.session_state.authenticated = True
                    st.session_state.api_key = st.text_input("Enter your Google API key", type="password")
                    st.success("Signed in successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

        elif choice == "Sign Up":
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Sign Up"):
                try:
                    user = auth.create_user(email=email, password=password)
                    st.success("User created successfully!")
                    user.send_email_verification()
                    st.info("Verification email sent. Please verify your email before signing in.")
                    st.session_state.authenticated = True
                    st.session_state.api_key = st.text_input("Enter your Google API key", type="password")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        enable_app(st.session_state.api_key)

def enable_app(api_key):
    # Add heading
    st.title("PDF VERSE AI")

    # Sidebar
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text, api_key)
                get_vector_store(text_chunks, api_key)
                st.success("Done")

    # Main area for asking questions
    st.subheader("Ask Any Question")

    with st.form(key="question_form"):
        user_question = st.text_input("Your Question")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_question:
        user_input(user_question, api_key)

if __name__ == "__main__":
    main()
