import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Solivox AI Assistant",
    page_icon="🤖", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B6BFF;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #6C757D;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .answer-container {
        background-color: #F0F7FF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4B6BFF;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-header'>🤖 Solivox AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Upload a PDF document and ask questions to get AI-powered answers</p>", unsafe_allow_html=True)

# Initialize session state for tracking processing status
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None

# Sidebar for API keys and configuration
with st.sidebar:
    st.image("SOLIVOX-AI.png", width=150)
    st.markdown("### Configuration")
    
    # For production, use st.secrets instead of hardcoded keys
    api_key = "AIzaSyCrt7aTKCag1U2t7vfutrODHvSacvyi6Ks"  # In production, use st.secrets["GOOGLE_API_KEY"]
    
    st.markdown("### About")
    st.info("""
    Solivox AI Assistant uses Google's Gemini model to analyze your PDF documents and 
    answer questions based on their content. Upload any PDF to get started!
    """)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
    
    # Process the uploaded file
    if uploaded_file:
        with st.spinner("Processing your document..."):
            # Create temp directory if it doesn't exist
            Path("temp").mkdir(exist_ok=True)
            pdf_path = f"temp/{uploaded_file.name}"
            
            # Save uploaded file
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the PDF
            loader = PyPDFLoader(pdf_path)
            text_splitter = CharacterTextSplitter(
                separator=".", 
                chunk_size=20000, 
                chunk_overlap=50
            )
            
            try:
                pages = loader.load_and_split(text_splitter)
                
                # Initialize embeddings and vector store
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key="AIzaSyCrt7aTKCag1U2t7vfutrODHvSacvyi6Ks"
                )
                
                # Create vector database directory if it doesn't exist
                #persist_directory = "./chromadb_local"
               # Path(persist_directory).mkdir(exist_ok=True)
                
                # Create and persist vector database
                vectordb = FAISS.from_documents(
                    pages, 
                    embedding=embeddings
                )
              #  vectordb.persist()
                
                # Store in session state
                st.session_state.vectordb = vectordb
                st.session_state.processing_complete = True
                
                st.success(f"✅ Successfully processed '{uploaded_file.name}'")
                
            except Exception as e:
                st.error(f"Error processing document: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Query section
    if st.session_state.processing_complete:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Ask Questions")
        query = st.text_input("What would you like to know about the document?")
        
        if query:
            with st.spinner("Finding answer..."):
                try:
                    # Initialize LLM
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash", 
                        api_key="AIzaSyCrt7aTKCag1U2t7vfutrODHvSacvyi6Ks"
                    )
                    
                    # Get retriever from vectordb
                    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 3})
                    
                    # Get relevant documents
                    documents = retriever.get_relevant_documents(query)
                    context_list = [doc.page_content for doc in documents]
                    
                    # Create prompt template
                    prompt = """You are Solivox AI Assistant. Provide a helpful, accurate, and concise answer based on the context provided.
                    
                    Context: {context}
                    
                    Question: {question}
                    
                    Answer:"""
                    
                    template = prompt.format(context=" ".join(context_list), question=query)
                    
                    # Get answer from LLM
                    answer = llm.invoke(template).content
                    
                    # Display answer
                    st.markdown("<div class='answer-container'>", unsafe_allow_html=True)
                    st.markdown("### 💡 Answer:")
                    st.markdown(answer)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Instructions and tips
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("### 📝 How It Works")
    st.markdown("""
    1. **Upload** your PDF document
    2. Wait for processing to **complete**
    3. **Ask** questions about the content
    4. Get AI-powered **answers** based on the document
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Tips for better questions
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("### 💡 Tips for Better Results")
    st.markdown("""
    - Ask specific questions for more precise answers
    - Provide context in your questions when possible
    - Try rephrasing if you don't get a satisfactory answer
    - Questions about facts, figures, and concepts in the document work best
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Document status
    if st.session_state.processing_complete:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("### 📊 Document Status")
        st.success("Document loaded and ready for queries")
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Powered by Google Gemini and Langchain | © 2025 Solivox AI")
