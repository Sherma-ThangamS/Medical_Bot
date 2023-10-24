import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from langchain.llms import GooglePalm

# Set environment variables and paths
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDp7w1aTllF9shGJGW8S8rcmiqVFJJh1KM'
DATA_PATH = 'data/'
DB_FAISS_PATH = 'db_faiss'

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Try to answer in points.

Context: {context}
Question: {question}

Return the answer below with the explanation in simple words with an example and deep knowledge.
Answer with explanation in simple words:
"""

# Define Streamlit app
st.title("Medical Chatbot")

# Create vector database (You may call this function when needed)
# create_vector_db()

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector store
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever(search_kwargs={"k": 5}),
                                                               memory=memory)
    return conversation_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = GooglePalm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []
# Define sidebar content
st.sidebar.title("Sample Questions")
st.sidebar.write("1. What are the symptoms of COVID-19?")
st.sidebar.write("2. How can I lower my blood pressure?")
st.sidebar.write("3. Tell me about diabetes management.")
st.sidebar.write("4. How to treat a common cold?")
st.sidebar.write("5. Describe the signs of a heart attack.")
st.sidebar.write("6. Contact Details:")
st.sidebar.write("   Email: sshermathangam.com")
def main():
    for message in st.session_state['conversation_history']:
        if message['role'] == 'user':
            st.text(f"User: {message['content']}")
        else:
            st.text(f"Bot: {message['content']}")

    # User input at the bottom
    user_query = st.text_input("Enter your medical query:")

    if st.button("Submit"):
        # Get the chatbot's response
        response = qa_bot()({'question': user_query})

        # Store the conversation history
        st.session_state['conversation_history'].append({"role": "user", "content": user_query})
        st.session_state['conversation_history'].append({"role": "assistant", "content": response['answer']})
        # Use st.empty() to update the response
        response_placeholder = st.empty()
        response_placeholder.text(f"Bot: {response['answer']}")


if __name__ == "__main__":
    main()
