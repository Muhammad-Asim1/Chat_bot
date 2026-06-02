import streamlit as st
import os
from vectorstore import process_all_pdfs_in_folder, get_vectorstore_stats
from rag_chain import chat_with_document, retrieve_only
from config import Config

# Set up the app
st.set_page_config(page_title="Document RAG System", page_icon="🤖", layout="wide")
st.title("🤖 Document RAG System")
st.markdown("*Hybrid search with AI-powered answers and grounded citations*")

# Create data folder if it doesn't exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.last_chunks = []  # Store chunks from last retrieval
    
    # Process all existing PDFs in data folder on startup
    with st.spinner("Processing existing PDFs in data/ folder..."):
        results = process_all_pdfs_in_folder(Config.UPLOAD_FOLDER)
        if "error" not in results and results.get("total_files", 0) > 0:
            st.session_state.startup_results = results

# Show sidebar with retrieved chunks at top, then settings below
with st.sidebar:
    # Initialize settings in session state
    if "mode" not in st.session_state:
        st.session_state.mode = "🤖 AI Generation (with citations)"
    
    # PRIORITY 1: Show retrieved chunks from last query (TOP OF SIDEBAR)
    if st.session_state.get("last_chunks"):
        st.header("🔍 Retrieved Chunks")
        st.caption(f"💬 {st.session_state.get('last_query', 'N/A')[:50]}..." if len(st.session_state.get('last_query', '')) > 50 else f"💬 {st.session_state.get('last_query', 'N/A')}")
        
        # Separate cited and non-cited chunks
        cited_chunks = [c for c in st.session_state.last_chunks if c.get('cited', False)]
        not_cited_chunks = [c for c in st.session_state.last_chunks if not c.get('cited', False)]
        
        # Show cited chunks first (if in AI generation mode)
        if cited_chunks:
            st.success(f"✅ {len(cited_chunks)} chunk(s) cited in answer")
            for chunk in cited_chunks:
                # Determine emoji based on freshness status
                freshness_status = chunk.get('freshness_status', 'unknown')
                if freshness_status == 'stale':
                    emoji = "⚠️"
                elif freshness_status == 'potentially_stale':
                    emoji = "⏳"
                else:
                    emoji = "✅"
                
                with st.expander(f"{emoji} Chunk {chunk['chunk_number']} • {chunk['score']:.0%}", expanded=False):
                    st.markdown(f"**Point ID:** `{chunk.get('point_id', 'N/A')}`")
                    st.markdown(f"**Source:** {chunk['source']}")
                    st.markdown(f"**Page:** {chunk['page']}")
                    
                    # Show freshness warning
                    if freshness_status == 'stale':
                        pub_date = chunk.get('publication_date', 'Unknown')
                        age_years = chunk.get('age_years', None)
                        age_str = f"{age_years:.1f} years old" if age_years else ""
                        st.warning(f"⚠️ **Stale Document** — Published {pub_date[:4] if pub_date else 'Unknown'} ({age_str}). May be outdated.")
                    elif freshness_status == 'potentially_stale':
                        pub_date = chunk.get('publication_date', 'Unknown')
                        age_years = chunk.get('age_years', None)
                        age_str = f"{age_years:.1f} years old" if age_years else ""
                        st.info(f"⏳ **Potentially Stale** — Published {pub_date[:4] if pub_date else 'Unknown'} ({age_str}). Verify current policy.")
                    if chunk.get('chunk_title'):
                        st.markdown(f"**Section:** {chunk['chunk_title']}")
                    st.markdown(f"**Relevance:** {chunk['score']:.2%}")
                    
                    # Show citation numbers that map to this chunk
                    citation_nums = chunk.get('citation_numbers', [])
                    if citation_nums:
                        citation_str = ', '.join([f'[{n}]' for n in sorted(citation_nums)])
                        st.success(f"✅ **Cited as:** {citation_str}")
                    
                    # Show highlight text(s) with their citation numbers
                    all_highlights = chunk.get('all_highlights', [])
                    if not all_highlights and chunk.get('highlight_text'):
                        all_highlights = [chunk.get('highlight_text')]
                    
                    if all_highlights:
                        st.divider()
                        if len(all_highlights) == 1:
                            cite_num = citation_nums[0] if citation_nums else ''
                            st.markdown(f"**🎯 Highlight [{cite_num}]:**")
                            st.info(f'"{all_highlights[0]}"')
                        else:
                            st.markdown(f"**🎯 {len(all_highlights)} Highlights:**")
                            for idx, highlight in enumerate(all_highlights):
                                cite_num = citation_nums[idx] if idx < len(citation_nums) else ''
                                st.info(f'[{cite_num}] "{highlight}"')
                    
                    st.divider()
                    st.markdown(f"*{chunk['preview']}*")
                    
                    # Add links for each citation number with its specific highlight
                    if chunk.get('source_url') and chunk['page'] != 'N/A':
                        citation_nums = chunk.get('citation_numbers', [])
                        all_highlights = chunk.get('all_highlights', [])
                        
                        if citation_nums and all_highlights:
                            # Create separate link for each citation number
                            for idx, cite_num in enumerate(citation_nums):
                                highlight = all_highlights[idx] if idx < len(all_highlights) else chunk.get('highlight_text', '')
                                if highlight:
                                    from rag_chain import encode_text_fragment
                                    encoded_text = encode_text_fragment(highlight)
                                    page_link = f"{chunk['source_url']}#page={chunk['page']}:~:text={encoded_text}"
                                    st.markdown(f"[📖 View [{cite_num}] in PDF]({page_link})")
                                else:
                                    page_link = f"{chunk['source_url']}#page={chunk['page']}"
                                    st.markdown(f"[📖 View [{cite_num}] in PDF]({page_link})")
                        elif chunk.get('highlight_text'):
                            # Single highlight
                            from rag_chain import encode_text_fragment
                            encoded_text = encode_text_fragment(chunk['highlight_text'])
                            page_link = f"{chunk['source_url']}#page={chunk['page']}:~:text={encoded_text}"
                            st.markdown(f"[📖 View in PDF (with highlight)]({page_link})")
                        else:
                            page_link = f"{chunk['source_url']}#page={chunk['page']}"
                            st.markdown(f"[📖 View in PDF]({page_link})")
    
        # Show not cited chunks
        if not_cited_chunks:
            with st.expander(f"📋 {len(not_cited_chunks)} additional chunk(s) retrieved (not cited)", expanded=False):
                for chunk in not_cited_chunks:
                    freshness_status = chunk.get('freshness_status', 'unknown')
                    freshness_emoji = "⚠️" if freshness_status == 'stale' else "⏳" if freshness_status == 'potentially_stale' else ""
                    st.markdown(f"**Chunk {chunk['chunk_number']}** • {chunk['score']:.0%} {freshness_emoji}")
                    st.caption(f"Point ID: `{chunk.get('point_id', 'N/A')}`")
                    st.markdown(f"📄 {chunk['source']}, Page {chunk['page']}")
                    
                    # Show freshness warning for stale/potentially stale
                    if freshness_status == 'stale':
                        pub_date = chunk.get('publication_date', 'Unknown')
                        st.caption(f"⚠️ Stale ({pub_date[:4] if pub_date else 'Unknown'})")
                    elif freshness_status == 'potentially_stale':
                        pub_date = chunk.get('publication_date', 'Unknown')
                        st.caption(f"⏳ Potentially stale ({pub_date[:4] if pub_date else 'Unknown'})")
                    
                    st.caption(chunk['preview'])
                    st.divider()
        
        st.divider()
    
    # PRIORITY 2: Settings (condensed in expander)
    with st.expander("⚙️ Settings", expanded=not st.session_state.get("last_chunks")):
        # Mode selector
        st.session_state.mode = st.radio(
            "Response Mode:",
            ["🤖 AI Generation (with citations)", "📄 Retrieval Only"],
            index=0 if st.session_state.mode == "🤖 AI Generation (with citations)" else 1,
            key="mode_radio"
        )
        
        # Strict Citation Mode toggle
        strict_mode = st.toggle(
            "🔒 Strict Citation Mode",
            value=Config.STRICT_CITATION_MODE,
            help="When ON: Only citations with verified exact text matches are shown. When OFF: Falls back to chunk position if verification fails.",
            key="strict_citation_toggle"
        )
        # Update Config dynamically
        Config.STRICT_CITATION_MODE = strict_mode

    # PRIORITY 3: Stats & Processing (condensed together)
    with st.expander("📊 System Info", expanded=False):
        st.caption("💡 Add PDFs to `data/` folder and restart to process them automatically")
        st.divider()
        
        stats = get_vectorstore_stats()
        
        if stats["exists"]:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Docs", stats["total_files"], label_visibility="visible")
            with col2:
                st.metric("📦 Chunks", stats["total_chunks"], label_visibility="visible")
    
            if stats["files"]:
                st.caption("**📁 Files:**")
                for file in stats["files"]:
                    st.caption(f"• {file}")
        else:
            st.info("No documents indexed")
        
        # Show startup processing results if any
        if "startup_results" in st.session_state:
            results = st.session_state.startup_results
            if results["newly_processed"] > 0 or results["already_processed"] > 0:
                st.caption("**🏁Startup:**")
                if results["newly_processed"] > 0:
                    st.caption(f"✅ {results['newly_processed']} new")
                if results["already_processed"] > 0:
                    st.caption(f"ℹ️ {results['already_processed']} cached")

# Initialize chat history and chunks if not set
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if query := st.chat_input("Ask a question about your documents..."):
    # Add user query to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Get response based on selected mode
    with st.spinner("🔍 Searching and generating answer..." if "AI Generation" in st.session_state.mode else "🔍 Retrieving relevant chunks..."):
        if "AI Generation" in st.session_state.mode:
            response, chunks_metadata = chat_with_document(query)
        else:
            response, chunks_metadata = retrieve_only(query)

    # Display response
    with st.chat_message("assistant"):
            st.markdown(response, unsafe_allow_html=True)
    
    # Store chunks metadata and query for sidebar display
    st.session_state.last_chunks = chunks_metadata
    st.session_state.last_query = query

    # Save response in chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update sidebar
    st.rerun()
