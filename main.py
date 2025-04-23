from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import List
import os
import re
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, JSONLoader,
    UnstructuredExcelLoader, UnstructuredPowerPointLoader
)
from PIL import Image
import pytesseract
from io import BytesIO
from dotenv import load_dotenv

# LangChain imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ===== Application Lifespan Management =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize heavy resources
    print("Initializing AI models...")
    app.state.llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )
    app.state.embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    app.state.session_store = {}
    yield
    # Cleanup (optional)
    print("Shutting down...")

# ===== FastAPI App Setup =====
app = FastAPI(lifespan=lifespan)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (your HTML UI)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ===== Health Check Endpoint =====
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "RAG API is running"}

# ===== Helper Functions =====
def parse_uploaded_files(uploaded_files: List[UploadFile]) -> List[Document]:
    documents = []
    for uploaded_file in uploaded_files:
        filename = uploaded_file.filename
        extension = filename.split(".")[-1].lower()
        contents = uploaded_file.file.read()

        temp_path = f"./temp.{extension}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        try:
            if extension == "pdf":
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
            elif extension == "csv":
                loader = CSVLoader(file_path=temp_path)
                docs = loader.load()
            elif extension == "json":
                loader = JSONLoader(file_path=temp_path)
                docs = loader.load()
            elif extension == "xlsx":
                loader = UnstructuredExcelLoader(temp_path)
                docs = loader.load()
            elif extension == "pptx":
                loader = UnstructuredPowerPointLoader(temp_path)
                docs = loader.load()
            elif extension in ["jpg", "jpeg", "png"]:
                image = Image.open(BytesIO(contents))
                text = pytesseract.image_to_string(image)
                docs = [Document(page_content=text, metadata={"source": filename})] if text.strip() else []
            else:
                continue
            documents.extend(docs)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return documents

def build_rag_chain(docs: List[Document], session_id: str):
    # Extract and scrape links
    links = set()
    for doc in docs:
        found_links = re.findall(r'https?://\S+', doc.page_content)
        links.update(found_links)

    headers = {"User-Agent": "Mozilla/5.0"}
    scraped_docs = []
    for url in links:
        try:
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            if text:
                scraped_docs.append(Document(page_content=text[:10000], metadata={"source": url}))
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")

    all_docs = docs + scraped_docs
    if not all_docs:
        raise HTTPException(status_code=400, detail="No valid text found in uploaded file(s).")

    # Process documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = splitter.split_documents(all_docs)
    vs = Chroma.from_documents(chunks, embedding=app.state.embeddings)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Setup conversation chain
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite user's question if needed based on chat history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use context to answer concisely:\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(app.state.llm, retriever, contextualize_q_prompt)
    qa_chain = create_stuff_documents_chain(app.state.llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    def get_history(session: str) -> BaseChatMessageHistory:
        if session not in app.state.session_store:
            app.state.session_store[session] = ChatMessageHistory()
        return app.state.session_store[session]

    return RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

# ===== Main API Endpoint =====
@app.post("/upload_and_query")
async def upload_and_query(
    query: str = Form(...),
    session_id: str = Form("default"),
    files: List[UploadFile] = File(...)
):
    try:
        documents = parse_uploaded_files(files)
        chain = build_rag_chain(documents, session_id)
        response = chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        return JSONResponse(content={"answer": response["answer"]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== Production Configuration =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=1,
        timeout_keep_alive=30
    )
