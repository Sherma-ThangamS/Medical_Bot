import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.ai.generativelanguage as glm
# from langchain.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from PIL import Image
import google.generativeai as genai
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder, speech_to_text

DATA_PATH = 'data/'
DB_FAISS_PATH = 'db_faiss'
api_key = st.secrets["API_KEY"]

# Function to create vector database
def create_vector_db():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    embeddings = GPT4AllEmbeddings()
    db = FAISS.from_documents(text_chunks, embeddings)
    db.save_local(DB_FAISS_PATH)

# Set custom prompt for the chatbot
def set_custom_prompt():
    prompt_template = """
    Use the following piece of context to answer the question. Use MUST provide a very detailed Answer atleast for each the question.

    {context}

    Question: {question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt

# Function to initialize the retrieval QA chain
def retrieval_qa_chain(llm, db):
    prompt = set_custom_prompt()
    chain_type = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type)
    return qa

# Function to initialize the chatbot
def qa_bot():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    llm = GoogleGenerativeAI(model="models/gemini-pro", convert_system_message_to_human=True, verbose=True, google_api_key=api_key)
    qa = retrieval_qa_chain(llm, db)
    return qa

# Initialize the conversation history
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

# Define sidebar content
st.sidebar.title("Sample Questions")
st.sidebar.write("1. Give some Basic Health Checklist?")
st.sidebar.write("2. I am having dry hair. Give tips to maintain it.")
st.sidebar.write("3. Tell me about diabetes management.")
st.sidebar.write("4. How to treat a common cold?")
st.sidebar.write("5. Describe the signs of a heart attack.")
st.sidebar.write("6. Contact Details:")
st.sidebar.write("   Email: sshermathangam.com")

# Main function to run the chatbot
def main():
    for message in st.session_state['conversation_history']:
        if message['role'] == 'user':
            st.markdown(f"*User:* {message['content']}")
        else:
            st.markdown(f"*Bot:* {message['content']}")

    # User input
    voice=""
    res=""
    columns = st.columns([3, 1])

# User input for medical query
    with columns[0]:
        user_query=st.text_input("Enter your medical query:")

    # Voice recording option
    with columns[1]:
        st.text("Voice input:")
        voice=speech_to_text("Start recording!",language='en', use_container_width=True, key='STT',)

    user_image = st.file_uploader(label="Image", type=['jpg', 'png'])

    if voice:
        st.text(voice)

    # st.write("Record your voice, and play the recorded audio:")
    # audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key='recorder')

    # if audio:
    #     st.audio(audio['bytes'])

    if voice:
        user_query=user_query+" "+voice
    print(user_query)
    if st.button("Submit"):
        retrieval_chain = qa_bot()

        # Process image if uploaded
        if user_image:
            img = Image.open(user_image)
            st.image(img, caption='Uploaded Image', use_column_width=True)
            bytes_data = user_image.getvalue()

            # Generate description using Generative AI
            model = genai.GenerativeModel('gemini-pro-vision')
            res = model.generate_content(
                glm.Content(
                    parts=[
                        glm.Part(text="Write a description about this picture. This description for the medical assistant who need the data from the image only."),
                        glm.Part(
                            inline_data=glm.Blob(
                                mime_type='image/jpeg',
                                data=bytes_data
                            )
                        ),
                    ],
                ),
                stream=True)
            res.resolve()
            res = res.text


        # try:
        #     response = retrieval_chain.run(user_query + res)
        # except:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key)
        response = retrieval_chain.run(res + "\n" + user_query)

        # Update conversation history
        st.session_state['conversation_history'].append({"role": "user", "content": user_query})
        st.session_state['conversation_history'].append({"role": "assistant", "content": response})

        # Display response
        response_placeholder = st.empty()
        response_placeholder.markdown(f"*Bot*: {response}")


if __name__ == "__main__":
    main()
