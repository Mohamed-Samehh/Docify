import streamlit as st
from document_processor import DocumentProcessor
from chatbot_service import ChatbotService
import os

# Page config
st.set_page_config(page_title="Document Chatbot", page_icon="📄", layout="wide")

@st.cache_resource
def get_document_processor():
    return DocumentProcessor()

@st.cache_resource
def get_chatbot_service(api_key):
    return ChatbotService(api_key)

@st.cache_data
def process_document(_processor, uploaded_file):
    chunks = _processor.load_from_upload(uploaded_file)
    return chunks

@st.cache_resource
def create_vectorstore(_processor, _chunks):
    return _processor.create_vectorstore(_chunks)

def main():
    st.title("📄 Document Chatbot")
    st.write("Upload a document to get summaries and ask questions using Groq + Kimi!")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Groq API Key", type="password")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
    
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar to continue.")
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
        try:
            with st.spinner("Processing document..."):
                chunks = process_document(doc_processor, uploaded_file)
                vectorstore = create_vectorstore(doc_processor, chunks)
            
            st.success(f"✅ Processed {len(chunks)} chunks from {uploaded_file.name}")
            
            # Store in session state
            st.session_state.chunks = chunks
            st.session_state.vectorstore = vectorstore
            st.session_state.doc_processor = doc_processor
            st.session_state.chatbot = chatbot
            
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
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            if question := st.chat_input("Ask a question about the document"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.write(question)
                
                # Generate response
                with st.chat_message("assistant"):
                    try:
                        # Search for relevant documents
                        relevant_docs = st.session_state.doc_processor.search_documents(
                            st.session_state.vectorstore, question, k=3
                        )
                        
                        # Stream response
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        for chunk in st.session_state.chatbot.answer_question(question, relevant_docs):
                            full_response += chunk
                            response_placeholder.write(full_response + "▌")
                        
                        response_placeholder.write(full_response)
                        
                        # Add assistant message
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response
                        })
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()