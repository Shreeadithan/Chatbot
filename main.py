## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (PyPDFLoader, CSVLoader, JSONLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import os
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## set up Streamlit 
st.title("Conversational RAG With PDF uplaods and chat history")
st.write("Upload files and chat with their content")

model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
)
llm = HuggingFacePipeline(pipeline=pipe)

## Check if groq api key is provided
if True:

    session_id=st.text_input("Session ID",value="default_session")
    ## statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files = st.file_uploader("Choose a file", type=["pdf", "csv", "json", "xlsx", "pptx", "jpg", "jpeg", "png"], accept_multiple_files=True)
    ## Process uploaded  PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            extension = filename.split(".")[-1].lower()
        
            filepath = f"./temp.{extension}"
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getvalue())

        if extension == "pdf":
            loader = PyPDFLoader(filepath)
            docs = loader.load()
        elif extension == "csv":
            loader = CSVLoader(file_path=filepath)
            docs = loader.load()
        elif extension == "json":
            loader = JSONLoader(file_path=filepath)
            docs = loader.load()
        elif extension == "xlsx":
            loader = UnstructuredExcelLoader(filepath)
            docs = loader.load()
        elif extension == "pptx":
            loader = UnstructuredPowerPointLoader(filepath)
            docs = loader.load()
        elif extension in ["jpg", "jpeg", "png"]:
            try:
                image = Image.open(filepath)
                text = pytesseract.image_to_string(image)
                from langchain.schema import Document 
                docs = [Document(page_content=text, metadata={"source": filename})]
            except Exception as e:
                st.error(f"❌ Failed to extract text from image: {e}")
                continue
        else:
            st.warning(f"Unsupported file type: {extension}")
            continue

        documents.extend(docs)

    # Split and create embeddings for the documents
        links = set()
        for doc in documents:
            found_links = re.findall(r'https?://\S+', doc.page_content)
            links.update(found_links)
        scraped_docs = []
        headers = {"User-Agent": "Mozilla/5.0"}
        for url in links:
            try:
                res = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(res.text, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                if text:
                    scraped_docs.append(Document(page_content=text[:10000], metadata={"source": url}))
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
        all_docs = documents + scraped_docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(all_docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()    

        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ## Answer question

        # Answer question
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")










