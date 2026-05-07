"""
Streamlit application for Product Review RAG System.
"""
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from product_rag_system import load_reviews, create_documents, create_vector_store, load_vector_store, query_product_with_sources

# Page configuration
st.set_page_config(
    page_title="Product Review Analyzer",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Product Review Analyzer")
st.markdown("Ask questions about your product reviews using AI")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# API Key input
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=os.getenv("OPENAI_API_KEY", ""),
    help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys"
)

# Validate and set API key
if not api_key:
    st.sidebar.warning("⚠️ Please enter your OpenAI API key to continue.")
    st.info("👈 Enter your OpenAI API key in the sidebar to get started.")
    st.stop()
else:
    # Set the API key in environment for the OpenAI client
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("✅ API key configured")

# Number of results slider
k_value = st.sidebar.slider("Number of reviews to retrieve", 3, 10, 5)

# Session state to track if data is loaded
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'doc_count' not in st.session_state:
    st.session_state.doc_count = 0
if 'products' not in st.session_state:
    st.session_state.products = []

# Load existing vector store and extract products on startup
if st.session_state.vectorstore is None and os.path.exists("./chroma_db"):
    try:
        st.session_state.vectorstore = load_vector_store()
        st.session_state.doc_count = st.session_state.vectorstore._collection.count()
        collection_data = st.session_state.vectorstore._collection.get(include=['metadatas'])
        if collection_data and collection_data.get('metadatas'):
            product_ids = set()
            for meta in collection_data['metadatas']:
                if meta and meta.get('product_id'):
                    product_ids.add(meta['product_id'])
            st.session_state.products = sorted(list(product_ids))
    except Exception:
        pass

# Product filter
st.sidebar.header("Product Filter")
if st.session_state.products:
    product_options = ["All Products"] + st.session_state.products
    selected_product = st.sidebar.selectbox(
        "Filter by product",
        product_options,
        help="Select a specific product to only search its reviews"
    )
else:
    selected_product = "All Products"
    st.sidebar.info("Upload data with ProductId column to enable filtering")

# Main content
st.header("📁 Step 1: Upload Reviews")

uploaded_file = st.file_uploader(
    "Upload a CSV file with product reviews",
    type=['csv'],
    help="The CSV should have a column named 'Text' or 'review_text' containing the reviews"
)

if uploaded_file is not None:
    # Load and preview data
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} reviews")

    # Show preview
    with st.expander("Preview data"):
        st.dataframe(df.head())

    # Find the text column
    text_column = None
    for col in ['Text', 'text', 'review_text', 'Review', 'review']:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        st.error("Could not find a text column. Please ensure your CSV has a column named 'Text' or 'review_text'")
    else:
        # Find rating column
        rating_column = None
        for col in ['Score', 'score', 'rating', 'Rating', 'stars']:
            if col in df.columns:
                rating_column = col
                break

        # Process button
        if st.button("🔄 Process Reviews", type="primary"):
            with st.spinner("Processing reviews... This may take a few minutes."):
                # Save temp file
                temp_path = "data/temp_upload.csv"
                df.to_csv(temp_path, index=False)

                # Clean and process
                clean_df = load_reviews(
                    temp_path,
                    text_column=text_column,
                    rating_column=rating_column if rating_column else 'Score',
                    max_reviews=5000  # Limit for performance
                )

                # Create documents
                documents = create_documents(clean_df)

                # Create vector store
                st.session_state.vectorstore = create_vector_store(documents)
                st.session_state.doc_count = len(documents)

                # Store unique products for filtering
                if 'product_id' in clean_df.columns:
                    st.session_state.products = clean_df['product_id'].dropna().unique().tolist()
                else:
                    st.session_state.products = []

                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            st.success(f"✅ Processed {st.session_state.doc_count} reviews!")

# Query section
st.header("❓ Step 2: Ask Questions")

if st.session_state.vectorstore is None:
    st.warning("👆 Please upload a CSV file to get started")
else:
    st.info(f"📂 Loaded database with {st.session_state.doc_count} reviews")

if st.session_state.vectorstore is not None:
    # Example questions
    st.markdown("**Example questions you can ask:**")
    example_questions = [
        "What do customers love most about this product?",
        "What are the main complaints?",
        "Is this product worth the price?",
        "What improvements do customers suggest?",
    ]

    cols = st.columns(2)
    for i, q in enumerate(example_questions):
        if cols[i % 2].button(q, key=f"example_{i}"):
            st.session_state.current_question = q

    # Question input
    question = st.text_input(
        "Your question:",
        value=st.session_state.get('current_question', ''),
        placeholder="e.g., What do customers think about the quality?"
    )

    if st.button("Search & Answer", type="primary") and question:
        with st.spinner("Searching reviews and generating answer..."):
            # Determine product filter
            product_filter = None if selected_product == "All Products" else selected_product

            result = query_product_with_sources(
                question,
                st.session_state.vectorstore,
                k=k_value,
                product_id=product_filter
            )

        # Display answer
        st.header("Answer")
        st.markdown(result['answer'])

        # Display sources
        st.header("Source Reviews")
        for i, source in enumerate(result['sources'], 1):
            # Build expander label with product info if available
            label = f"Review {i} - Rating: {source['rating']}/5 | Sentiment: {source['sentiment']}"
            if source.get('product_id'):
                label += f" | Product: {source['product_id']}"
            label += f" | Similarity: {source['similarity_score']:.2%}"

            with st.expander(label):
                st.write(source['text'])

# Footer
st.markdown("---")
st.markdown(
    "Built using LangChain, ChromaDB, and OpenAI | "
)
