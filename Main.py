import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PIL import Image
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder, speech_to_text
from langchain_community.embeddings import HuggingFaceEmbeddings, GPT4AllEmbeddings
from langchain.chat_models import ChatOpenAI
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


DATA_PATH = 'data/'
DB_FAISS_PATH = 'db_faiss'
api_key = st.secrets["API_KEY"]

# Function to create vector database
def create_vector_db():
    st.info("Loading and processing PDFs...")

    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()

    # Filter out pages with no text or junk content
    cleaned_docs = []
    for doc in documents:
        if doc.page_content and len(doc.page_content.strip()) > 30:  # filter out blank or nearly empty
            cleaned_docs.append(doc)

    # Log how many valid docs were kept
    st.success(f"Loaded {len(cleaned_docs)} clean pages from PDF.")

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(cleaned_docs)

    # Use more general-purpose embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )

    # Create and save vector store
    db = FAISS.from_documents(text_chunks, embeddings)
    db.save_local(DB_FAISS_PATH)

    st.success("✅ Vector DB created and saved successfully.")

def generate_caption(image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(caption)
    return caption


# Set custom prompt for the chatbot
def set_custom_prompt():
    prompt_template = """
    Use the following context to answer the question. If the context is not relevant or lacks information, politely say you don't know.

    Context:
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
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    retriever = db.as_retriever()

    # Use OpenRouter-compatible ChatOpenAI (e.g. Claude)
    llm = ChatOpenAI(
        model="mistralai/mistral-7b-instruct",  # You can change to mistral, llama3, etc.
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        temperature=0.7,
        max_tokens=2000
    )
    
    
    prompt=set_custom_prompt();
    
    

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

    return qa

# Initialize the conversation history
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

# Sidebar content
st.sidebar.title("Sample Questions")
st.sidebar.write("1. Give some Basic Health Checklist?")
st.sidebar.write("2. I am having dry hair. Give tips to maintain it.")
st.sidebar.write("3. Tell me about diabetes management.")
st.sidebar.write("4. How to treat a common cold?")
st.sidebar.write("5. Describe the signs of a heart attack.")
st.sidebar.write("6. Contact Details:")
st.sidebar.write("   Email: sshermathangam.com")

if st.sidebar.button("Rebuild Vector DB"):
    create_vector_db()

# Main function to run the chatbot
def main():
    for message in st.session_state['conversation_history']:
        role = message['role']
        st.markdown(f"*{role.capitalize()}*: {message['content']}")

    voice = ""
    res = ""
    columns = st.columns([3, 1])

    with columns[0]:
        user_query = st.text_input("Enter your medical query:")

    with columns[1]:
        st.text("Voice input:")
        voice = speech_to_text("Start recording!", language='en', use_container_width=True, key='STT')

    user_image = st.file_uploader(label="Image", type=['jpg', 'png'])

    if voice:
        st.text(voice)
        user_query += " " + voice

    if st.button("Submit"):
        retrieval_chain = qa_bot()

        # Image processing (mock response since Gemini is removed)
        if user_image:
            img = Image.open(user_image)
            st.image(img, caption='Uploaded Image', use_column_width=True)
            caption = generate_caption(img)
            res = f"Image Caption: {caption}. The uploaded image is noted for reference."

        # Run the retrieval chain
        try:
            answer = retrieval_chain(res + "\n" + user_query)
            response = answer["result"]
            sources = answer["source_documents"]

            # Print or display top source document for debugging
            if sources:
                st.markdown("**Top Source Context (for debug):**")
                st.code(sources[0].page_content[:1000])  # display a preview

        except Exception as e:
            response = f"❌ Error: {e}"

        # Update chat history
        st.session_state['conversation_history'].append({"role": "user", "content": user_query})
        st.session_state['conversation_history'].append({"role": "assistant", "content": response})

        # Display result
        st.markdown(f"*Bot*: {response}")

if __name__ == "__main__":
    main()
