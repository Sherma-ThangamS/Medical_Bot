import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI ,GoogleGenerativeAI
import os
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai


DATA_PATH = 'data/'
DB_FAISS_PATH = 'db_faiss'

# Create vector database
def create_vector_db():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    embeddings=GPT4AllEmbeddings()
    db = FAISS.from_documents(text_chunks, embeddings)
    db.save_local(DB_FAISS_PATH)

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Try to answer in points.

Context: {context}
Question: {input}

Return the answer below with the explanation in simple words with an example and deep knowledge.
Answer with explanation in simple words:
"""

# Define Streamlit app
st.title("Medical Chatbot")

# Create vector database (You may call this function when needed)
# create_vector_db()

def set_custom_prompt():
    prompt_template  = """
Use the following piece of context to answer the question. Use MUST provide a very detailed Answer atleast for each the question.

{context}

Question: {question}
"""
    prompt = PromptTemplate(template = prompt_template , input_variables=["context", "question"])
    return prompt

def retrieval_qa_chain(llm, db):
    retriever=db.as_retriever()
    prompt=set_custom_prompt()
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    return qa

def qa_bot():
    embeddings=GPT4AllEmbeddings()
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = GoogleGenerativeAI(model="models/text-bison-001",convert_system_message_to_human=True,verbose=True,google_api_key="AIzaSyAtp_lUKFAXhp9O1B_nmg_pvWGAuVxaXZ8")

    qa = retrieval_qa_chain(llm, db)
    return qa
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []
# Define sidebar content
st.sidebar.title("Sample Questions")
st.sidebar.write("1. Give some Basic Health Checklist?")
st.sidebar.write("2. I having dry hair, Give tips to maintain it.")
st.sidebar.write("3. Tell me about diabetes management.")
st.sidebar.write("4. How to treat a common cold?")
st.sidebar.write("5. Describe the signs of a heart attack.")
st.sidebar.write("6. Contact Details:")
st.sidebar.write("   Email: sshermathangam.com")
def main():
    for message in st.session_state['conversation_history']:
        if message['role'] == 'user':
            st.markdown(f"User: {message['content']}")
        else:
            st.markdown(f"Bot: {message['content']}")

    # User input at the bottom
    user_query = st.text_input("Enter your medical query:")
    user_image=st.file_uploader(label="Image")
    if st.button("Submit"):
        retrieval_chain = qa_bot()
        try:
            response = retrieval_chain.run(user_query)
        except:
            # llm = GoogleGenerativeAI(model="models/text-bison-001",convert_system_message_to_human=True,verbose=True,google_api_key="AIzaSyAtp_lUKFAXhp9O1B_nmg_pvWGAuVxaXZ8")
            # response=llm.generate([user_query], output_format="text", max_length=256, num_beams=5, length_penalty=0.8)
            response="Out of my context!!"
        if user_image:
            img=Image.open(user_image)
            st.image(img)
        st.session_state['conversation_history'].append({"role": "user", "content": user_query})
        st.session_state['conversation_history'].append({"role": "assistant", "content":response})
        # Use st.empty() to update the response
        response_placeholder = st.empty()
        response_placeholder.markdown(f"Bot: {response}")


if __name__ == "__main__":
    main()
