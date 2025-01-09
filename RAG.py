import streamlit as st
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
import os
import tempfile
from typing import List, Dict, Any
import time
from datetime import datetime, timedelta
import threading
from queue import Queue
import plotly.graph_objects as go
from collections import deque
import tiktoken
import asyncio

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()

    def can_proceed(self) -> bool:
        """Check if request can proceed under rate limits."""
        now = datetime.now()
        with self.lock:
            # Remove old requests
            while self.requests and self.requests[0] < now - timedelta(seconds=self.time_window):
                self.requests.popleft()
            
            # Check if we can make a new request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.placeholder = container.empty()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.placeholder.markdown(self.text)

class DocumentProcessor:
    def __init__(self, openai_api_key: str):
        """Initialize document processor with OpenAI API key."""
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            streaming=True
        )
        self.vector_store = None
        self.chunks = []
        self.rate_limiter = RateLimiter(max_requests=20, time_window=60)  # 20 requests per minute
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def visualize_chunks(self) -> go.Figure:
        """Create a visualization of document chunks."""
        chunk_sizes = [self.count_tokens(chunk.page_content) for chunk in self.chunks]
        chunk_numbers = list(range(1, len(chunk_sizes) + 1))
        
        fig = go.Figure(data=[
            go.Bar(
                x=chunk_numbers,
                y=chunk_sizes,
                text=chunk_sizes,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Document Chunks Token Distribution",
            xaxis_title="Chunk Number",
            yaxis_title="Token Count",
            showlegend=False
        )
        
        return fig

    def process_file(self, uploaded_file) -> bool:
        """Process uploaded file and create vector store."""
        try:
            # Create a temporary file to store the uploaded content
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            # Load and process the document based on file type
            if uploaded_file.name.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif uploaded_file.name.lower().endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                raise ValueError("Unsupported file type")

            documents = loader.load()
            self.chunks = self.text_splitter.split_documents(documents)
            
            # Calculate and display token counts
            total_tokens = sum(self.count_tokens(chunk.page_content) for chunk in self.chunks)
            st.sidebar.write(f"Total chunks: {len(self.chunks)}")
            st.sidebar.write(f"Total tokens: {total_tokens}")
            
            # Visualize chunks
            st.sidebar.plotly_chart(self.visualize_chunks(), use_container_width=True)
            
            self.vector_store = FAISS.from_documents(self.chunks, self.embeddings)
            
            # Clean up temporary file
            os.unlink(file_path)
            return True
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return False

    async def query_document(self, query: str, stream_container: st.delta_generator.DeltaGenerator) -> Dict:
        """Query the processed document with streaming response."""
        if not self.vector_store:
            raise ValueError("No document has been processed yet")

        if not self.rate_limiter.can_proceed():
            raise Exception("Rate limit exceeded. Please wait before making more requests.")

        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True
            )

            stream_handler = StreamHandler(stream_container)
            response = await qa_chain.ainvoke(
                {"query": query},
                callbacks=[stream_handler]
            )

            return {
                "answer": response["result"],
                "sources": [doc.page_content for doc in response["source_documents"]]
            }
        except Exception as e:
            error_msg = str(e)
            if "rate limits" in error_msg.lower():
                raise Exception("OpenAI rate limit reached. Please wait a moment before trying again.")
            raise

def initialize_session_state():
    """Initialize session state variables."""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0

def main():
    st.title("📚 Mini Notebook LM")
    st.write("Upload a document and ask questions about its content!")

    # Initialize session state
    initialize_session_state()

    # Sidebar for API key and file upload
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        
        uploaded_file = st.file_uploader(
            "Upload Document (PDF or TXT)",
            type=["pdf", "txt"],
            key="file_uploader"
        )

        if uploaded_file and uploaded_file != st.session_state.current_file:
            if api_key:
                try:
                    st.session_state.current_file = uploaded_file
                    with st.spinner("Initializing processor..."):
                        st.session_state.processor = DocumentProcessor(api_key)
                        st.session_state.is_processing = True
                        
                        with st.spinner("Processing document..."):
                            if st.session_state.processor.process_file(uploaded_file):
                                st.success("Document processed successfully!")
                                st.session_state.chat_history = []
                                st.session_state.error_count = 0
                            st.session_state.is_processing = False
                except Exception as e:
                    st.error(f"Error initializing processor: {str(e)}")
                    st.session_state.processor = None
                    st.session_state.is_processing = False
                    st.session_state.error_count += 1
            else:
                st.error("Please enter your OpenAI API key first")

        if st.button("Clear Session", use_container_width=True):
            initialize_session_state()
            st.rerun()

        # Display error count if any
        if st.session_state.error_count > 0:
            st.sidebar.error(f"Errors encountered: {st.session_state.error_count}")

    # Main content area
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to begin")
        return

    if not uploaded_file:
        st.info("Please upload a document in the sidebar to begin")
        return

    # Query input
    query = st.text_input("Ask a question about your document:", key="query_input")
    if query and st.session_state.processor and not st.session_state.is_processing:
        try:
            # Create a container for streaming response
            stream_container = st.empty()
            
            with st.spinner("Processing query..."):
                response = asyncio.run(st.session_state.processor.query_document(query, stream_container))
                
                st.session_state.chat_history.append({
                    "query": query,
                    "response": response["answer"],
                    "timestamp": time.strftime("%H:%M:%S"),
                    "sources": response["sources"]
                })

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.session_state.error_count += 1

    # Display chat history with expandable sources
    if st.session_state.chat_history:
        st.header("Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.write(f"**Time:** {chat['timestamp']}")
                st.write("**Question:**")
                st.write(chat["query"])
                st.write("**Answer:**")
                st.write(chat["response"])
                
                # Expandable sources section
                with st.expander("View Sources"):
                    for j, source in enumerate(chat["sources"], 1):
                        st.write(f"Source {j}:")
                        st.write(source)
                
                st.divider()

if __name__ == "__main__":
    main()