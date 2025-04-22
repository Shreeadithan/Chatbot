# ⚡RAGnarok Chatbot ⚡
**RAGnarok** is a Retrieval-Augmented Generation (RAG) chatbot application designed to let users upload and interact with a wide variety of document types—including **PDF, DOCX, PPTX, XLSX, PNG, JPG, CSV, JSON, and TXT**—by asking natural language questions and receiving accurate, context-aware answers based on the content of those files.
Absolutely! Here’s an updated introduction that highlights the chat message history feature along with the other requested points:

- RAGnarok solves the problem of information overload and inefficient document search by enabling instant, conversational access to the knowledge buried within your files. Its primary use cases include business document search, academic research, legal discovery, technical support, and any scenario where quick, accurate answers from diverse documents are needed.  
- RAGnarok is ideal for professionals, researchers, students, and teams who regularly work with large volumes of information across multiple formats.  
- What makes RAGnarok unique is its seamless combination of multi-format document ingestion, semantic search, chat message history, and advanced language models—delivering trustworthy, context-aware answers grounded in your own data, without the need for retraining or manual indexing.
- Its persistent chat history allows users to maintain context across conversations, revisit previous queries, and build on earlier discussions for a more natural and productive user experience. With its user-friendly interface and flexible deployment options, RAGnarok transforms the way you access and utilize your knowledge base.
- This RAG chatbot was developed as a submission for the Mando hackathon, which encourages innovative AI solutions for enterprise productivity and support.
- The chatbot aligns with Mando’s mission to empower enterprise software users—such as admins and IT professionals—by enabling instant, AI-powered access to information across diverse document types.

## WHAT IS RAG?
Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. **Large Language Models (LLMs)** are trained on vast volumes of data and use billions of parameters to generate original output for tasks like answering questions, translating languages, and completing sentences

## PROJECT STRUCTURE
Here is an overview of project's organization

    ├── requirements.txt        # contains the list of packages required to run the codes given
    ├── .env                    # stores environment variables for an application
    ├── frontend.html           # creates the user interface of a website or web application
    ├── main.py                 # primary code file containing the code for developing a RAG application
    └── README.md               # Guidance for readers to comprehend

## KEY FEATURES AND CAPABILITIES
- **Multi-source knowledge integration:** Ability to process and retrieve information from PDFs, web pages, databases, etc.
- **Contextual understanding:** Advanced semantic search capabilities for retrieving the most relevant information
- **Customizable prompt templates:** Flexibility to adjust system prompts for different use cases
- **Citation tracking:** Automatic tracking of source documents used in responses
- **Conversation memory:** Support for multi-turn conversations with context retention

## ARCHITECTURE AND COMPONENTS

### CORE ARCHITECTURE AND DESIGN
![alt text](https://github.com/Shreeadithan/Chatbot/blob/main/RAGNAROK.webp)

### RAG PIPELINE COMPONENTS

#### 1. 📥 Data Ingestion and Preprocessing
- Supported formats: PDF, CSV, JSON, PPTX, XLSX, images
- Parsing tools: LangChain loaders + BeautifulSoup (for scraping)
- Metadata extraction: Captures file name, page number, URL
- Chunking strategy: RecursiveCharacterTextSplitter
-     `chunk_size = 5000`, `chunk_overlap = 500`
  
#### 2. 🧬 Embedding and Indexing
- Embedding model: all-MiniLM-L6-v2 from HuggingFace (compact & fast)
- Vector database: Chroma DB (local, lightweight, fast)
- Indexing strategy: Semantic vector storage per chunk
- Storage: In-memory Chroma, optionally persisten

#### 3. 🔍 Retrieval Mechanism
- Preprocessing: Optional query rephrasing using conversation history
- Retrieval: Vector similarity search using Chroma's retriever
- Ranking: Based on cosine similarity of embedded query vs document chunks
- Top-k results: Default retrieves top 3 most relevant chunks

#### 4. 🧩 Context Augmentation
- Formatting: Injects context into the prompt in natural readable form
- Window management: Handles token limitations via chunk size and overlap
- Prioritization: Retains top-ranked, highly relevant chunks
- Metadata preservation: Maintains source references for citation

#### 5. 🧠 Generation
- LLM used: LLaMA 3.3 70B (via Groq API)
- Prompting framework: ChatPromptTemplate with history and context placeholders

## ⚙️ Technologies, Libraries, and Frameworks

| Tool/Library          | Purpose                                           |
| --------------------- | ------------------------------------------------- |
| FastAPI               | Backend API framework for file upload & querying  |
| Uvicorn               | ASGI server for serving FastAPI                   |
| LangChain             | Building blocks for RAG pipelines                 |
| Chroma                | Lightweight vector database                       |
| sentence-transformers | For generating text embeddings                    |
| pytesseract + Pillow  | OCR for extracting text from images               |
| pdfminer.six, PyPDF2  | PDF parsing utilities                             |
| BeautifulSoup4        | Scraping and parsing webpage text                 | 
| dotenv                | Environment variable management for API keys      |
| HTML/CSS/JS           | Frontend user interface with feedback capabilities|

## INSTALLATION INSTRUCTIONS
```
# 1. Clone the repository
git clone https://github.com/Shreeadithan/Chatbot.git
cd your-rag-app

# 2. Create and activate a virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# (Then open .env and replace placeholders with your actual API keys)

# 5. Run the backend server
uvicorn backend:app --reload

# 6. Open the frontend
# Option 1: Double-click index.html to open in browser (best for local)
# Option 2: Serve via Python if needed:
python -m http.server 3000
# Then go to: http://localhost:3000
```

