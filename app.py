import streamlit as st
from document_processor import DocumentProcessor
from chatbot_service import ChatbotService
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Docify", page_icon="📄", layout="wide")

@st.cache_resource(show_spinner="Initializing document processor...")
def get_document_processor():
    return DocumentProcessor()

@st.cache_resource(show_spinner="Setting up AI chatbot service...")
def get_chatbot_service(api_key):
    return ChatbotService(api_key)

@st.cache_data(show_spinner="Analyzing document structure...")
def process_document(_processor, file_content, file_name, _timestamp):
    """Process document with content-based caching and timestamp for uniqueness"""
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

@st.cache_data(show_spinner="Building searchable index...")
def create_vectorstore(_processor, _chunks, _timestamp):
    """Create vectorstore with timestamp for cache uniqueness"""
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
            # Clear previous session state and caches for new file
            process_document.clear()
            create_vectorstore.clear()
            keys_to_clear = ['chunks', 'vectorstore', 'messages', 'current_file', 'summary_generated', 'summary_content']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
        
        try:
            with st.spinner("Processing your document - this may take a moment..."):
                # Get file content and name for better caching
                file_content = uploaded_file.getvalue()
                file_name = uploaded_file.name
                
                # Add timestamp to ensure unique cache keys for each upload session
                import time
                timestamp = time.time()
                
                chunks = process_document(doc_processor, file_content, file_name, timestamp)
                vectorstore = create_vectorstore(doc_processor, chunks, timestamp)
            
            # Store in session state
            st.session_state.chunks = chunks
            st.session_state.vectorstore = vectorstore
            st.session_state.doc_processor = doc_processor
            st.session_state.chatbot = chatbot
            st.session_state.current_file = uploaded_file.name  # Track current file
            st.session_state.file_timestamp = timestamp  # Track when file was processed
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return
    else:
        # File was removed - clear all related session state AND clear caches
        if 'chunks' in st.session_state:
            # Clear caches to prevent old data from persisting
            process_document.clear()
            create_vectorstore.clear()
            
            keys_to_clear = ['chunks', 'vectorstore', 'messages', 'current_file', 'summary_generated', 'summary_content', 'doc_processor', 'chatbot']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Only show functionality if document is processed
    if 'chunks' in st.session_state:
        # Initialize states
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "summary_generated" not in st.session_state:
            st.session_state.summary_generated = False
        if "summary_content" not in st.session_state:
            st.session_state.summary_content = ""
            
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📋 Summary")
            if st.button("Generate Summary", type="primary", disabled=st.session_state.get('generating_summary', False)):
                st.session_state.generating_summary = True
                with st.spinner("Analyzing document and creating summary..."):
                    try:
                        summary = st.session_state.chatbot.summarize_document(
                            st.session_state.chunks
                        )
                        st.session_state.summary_content = summary
                        st.session_state.summary_generated = True
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
                    finally:
                        st.session_state.generating_summary = False
                        st.rerun()
            
            # Show summary if generated
            if st.session_state.summary_generated and st.session_state.summary_content:
                st.write(st.session_state.summary_content)
        
        with col2:
            st.header("💬 Ask Questions")
            
            # Chat input at the top
            question = st.chat_input("Ask a question about the document")
            
            # Display chat history with thinking indicator in the right position
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                
                # Show thinking indicator right after the first (newest) user message if processing
                if (i == 0 and message["role"] == "user" and 
                    st.session_state.get('processing_question', False)):
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            # Process the pending question while showing the spinner
                            user_question = message["content"]
                            
                            try:
                                # Search for relevant documents
                                relevant_docs = st.session_state.doc_processor.search_documents(
                                    st.session_state.vectorstore, user_question, k=3
                                )
                                
                                # Generate response
                                full_response = ""
                                for chunk in st.session_state.chatbot.answer_question(user_question, relevant_docs):
                                    full_response += chunk
                                
                                # Add assistant response right after user message (at position 1)
                                st.session_state.messages.insert(1, {"role": "assistant", "content": full_response})
                                
                            except Exception as e:
                                st.session_state.messages.insert(1, {"role": "assistant", "content": f"Error generating response: {str(e)}"})
                            
                            finally:
                                st.session_state.processing_question = False
                                st.rerun()
            
            if question:
                # Check if we still have a valid document
                if 'vectorstore' not in st.session_state or 'chunks' not in st.session_state:
                    st.error("Please upload a document first.")
                    return
                
                # Prevent multiple concurrent requests
                if st.session_state.get('processing_question', False):
                    return
                    
                # Add user message first and set processing state
                st.session_state.messages.insert(0, {"role": "user", "content": question})
                st.session_state.processing_question = True
                st.rerun()  # Refresh to show user message and thinking indicator

if __name__ == "__main__":
    main()