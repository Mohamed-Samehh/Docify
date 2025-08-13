import streamlit as st
from document_processor import DocumentProcessor
from chatbot_service import ChatbotService
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Docify", page_icon="📄", layout="wide")

@st.cache_resource
def get_document_processor():
    return DocumentProcessor()

@st.cache_resource
def get_chatbot_service(api_key):
    return ChatbotService(api_key)

@st.cache_data
def process_document(_processor, file_content, file_name):
    """Process document with content-based caching"""
    import tempfile
    import os
    
    # Create a temporary file with the uploaded content
    with tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=f".{file_name.split('.')[-1]}"
    ) as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name
    
    try:
        chunks = _processor.load_document(tmp_file_path)
        return chunks
    finally:
        os.unlink(tmp_file_path)

@st.cache_data
def create_vectorstore(_processor, _chunks):
    return _processor.create_vectorstore(_chunks)

def main():
    st.title("📄 Docify (Document Chatbot)")
    st.write("Upload a document to get summaries and ask questions!")
    
    # Get API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.warning("Please set GROQ_API_KEY in your .env file to continue.")
        return
    
    # Initialize services
    doc_processor = get_document_processor()
    chatbot = get_chatbot_service(api_key)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['pdf', 'txt'],
        help="Upload a PDF or text file"
    )
    
    if uploaded_file:
        # Check if this is a different file than previously processed
        if ('current_file' in st.session_state and 
            st.session_state.current_file != uploaded_file.name):
            # Clear previous session state for new file
            keys_to_clear = ['chunks', 'vectorstore', 'messages', 'current_file']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
        
        try:
            with st.spinner("Processing document..."):
                # Get file content and name for better caching
                file_content = uploaded_file.getvalue()
                file_name = uploaded_file.name
                
                chunks = process_document(doc_processor, file_content, file_name)
                vectorstore = create_vectorstore(doc_processor, chunks)
            
            # Store in session state
            st.session_state.chunks = chunks
            st.session_state.vectorstore = vectorstore
            st.session_state.doc_processor = doc_processor
            st.session_state.chatbot = chatbot
            st.session_state.current_file = uploaded_file.name  # Track current file
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return
    
    # Only show functionality if document is processed
    if 'chunks' in st.session_state:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📋 Summary")
            if st.button("Generate Summary", type="primary"):
                with st.spinner("Generating summary..."):
                    try:
                        summary = st.session_state.chatbot.summarize_document(
                            st.session_state.chunks
                        )
                        st.write(summary)
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
        
        with col2:
            st.header("💬 Ask Questions")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Chat input at the top
            if question := st.chat_input("Ask a question about the document"):
                # Generate response first
                try:
                    # Search for relevant documents
                    relevant_docs = st.session_state.doc_processor.search_documents(
                        st.session_state.vectorstore, question, k=3
                    )
                    
                    # Get full response before adding to messages
                    full_response = ""
                    for chunk in st.session_state.chatbot.answer_question(question, relevant_docs):
                        full_response += chunk
                    
                    # Add both messages to the beginning of the list
                    st.session_state.messages.insert(0, {"role": "assistant", "content": full_response})
                    st.session_state.messages.insert(0, {"role": "user", "content": question})
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
            
            # Display chat history (newest first)
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

if __name__ == "__main__":
    main()