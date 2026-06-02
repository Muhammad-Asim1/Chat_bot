# 📄 NUST Policy RAG System

A production-grade RAG (Retrieval-Augmented Generation) system for querying NUST policy documents with hybrid search, AI-powered answers, and grounded citations.

## 🌟 Features

- **🤖 AI-Powered Answers**: Groq (Llama 3.1) generates natural responses from retrieved context
- **🎯 Hybrid Search**: Dense embeddings (Jina) + Sparse BM25 (Qdrant) + Reranking (Jina)
- **📚 Grounded Citations**: Clickable links to source PDFs with page-level deep linking
- **⚡ All-Cloud Processing**: Fast performance without local model loading
- **📊 Comprehensive Evaluation**: Ragas metrics, nDCG, AIS, and abstention evaluation
- **🧪 Test Suite**: Unit tests for chunking, retrieval, and citations

## 🚀 Quick Start

### 1. Setup Conda Environment

```bash
# Create a new conda environment
conda create -n nust_policy python=3.10 -y

# Activate the environment
conda activate nust_policy

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (Required)
JINA_API_KEY=your_jina_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Qdrant Configuration (Optional - defaults shown)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=hybrid-search
```

**Get your API keys:**
- **Jina**: [jina.ai](https://jina.ai) - For embeddings and reranking
- **Groq**: [console.groq.com](https://console.groq.com) - For LLM inference
- **Google**: [makersuite.google.com](https://makersuite.google.com/app/apikey) - For Gemini (used in Ragas evaluation)

### 3. Start Qdrant Vector Database

```bash
# Build and start Qdrant container
docker-compose up --build -d

# Verify Qdrant is running
curl http://localhost:6333/health
```

The Qdrant dashboard will be available at: http://localhost:6333/dashboard

### 4. Run Streamlit Application

```bash
# Make sure conda environment is activated
conda activate nust_policy

# Start the Streamlit app
streamlit run app.py
```

The app will:
- Automatically ingest all PDFs from the `data/` folder
- Process documents with hybrid embeddings (Dense + Sparse)
- Open at http://localhost:8501

**Note**: On first run, the app processes all PDFs in `data/`. This may take a few minutes .

## 📁 Project Structure

```
├── app.py                    # Streamlit web interface
├── rag_chain.py              # RAG pipeline (retrieval + generation)
├── vectorstore.py            # Document ingestion and vector store management
├── config.py                 # Configuration settings
├── clear_index.py            # Utility to clear Qdrant index
├── requirements.txt           # Python dependencies
├── docker-compose.yml        # Qdrant container configuration
├── data/                     # PDF documents (auto-processed on startup)
├── evaluation/
│   ├── eval.py               # Main evaluation script
│   └── dataset_eval.py        # Evaluation questions and ground truth
├── tests/                    # Test suite
│   ├── test_chunker.py        # Chunking tests
│   ├── test_retriever.py      # Retrieval tests
│   └── test_citations.py     # Citation tests
└── qdrant_data/              # Vector database storage (Docker volume)
```

## ⚙️ Configuration

Edit `config.py` to customize system behavior:

### Retrieval Parameters

```python
PREFETCH_LIMIT = 10      # Results from each sub-query (dense + sparse)
RERANK_LIMIT = 6         # Number of candidates to send to reranker
FINAL_LIMIT = 3          # Final number of results after reranking
```

### Chunking Settings

```python
CHUNK_SIZE = 1200        # Characters per chunk
CHUNK_OVERLAP = 250      # Overlap between chunks
CHUNKING_STRATEGY = "fixed"  # or "semantic"
```

### Abstention Threshold

```python
MIN_RELEVANCE_SCORE = 0.30  # Minimum rerank score to answer (0-1 scale)
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_retriever.py

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

**Test Files:**
- `test_chunker.py`: Tests document chunking logic
- `test_retriever.py`: Tests hybrid retrieval pipeline
- `test_citations.py`: Tests citation extraction and validation

## 📊 Running Evaluation

The evaluation system computes multiple metrics:
- **Ragas Metrics**: faithfulness, answer_relevancy, context_precision, context_recall
- **Retrieval Metrics**: nDCG, Recall@k
- **Citation Metrics**: Citation Precision, AIS Recall, AIS F1
- **Abstention Metrics**: Abstention Rate, Recall, False Answer Rate

### Basic Evaluation

```bash
# Run full evaluation on all questions
python evaluation/eval.py

# Run evaluation on specific question range
python evaluation/eval.py --start 0 --end 50

# Add delay between API calls to avoid rate limits
python evaluation/eval.py --delay 2.0
```

### Ragas-Only Evaluation

If you have an intermediate CSV file with generated answers:

```bash
# Run Ragas evaluation on existing intermediate file
python evaluation/eval.py --ragas-only ragas_eval_results_intermediate.csv

# With custom batch size and delay
python evaluation/eval.py --ragas-only ragas_eval_results_intermediate.csv \
    --batch-size 5 --batch-delay 10
```

### Abstention Evaluation

```bash
# Run abstention evaluation
python evaluation/eval.py --abstention

# With custom range and delay
python evaluation/eval.py --abstention --start 0 --end 20 --delay 1.0
```

### Evaluation Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--start` | Starting question index (0-based) | 0 |
| `--end` | Ending question index (exclusive) | None (all) |
| `--save` | Output CSV path | `ragas_eval_results.csv` |
| `--delay` | Delay between API calls (seconds) | 0 |
| `--abstention` | Run abstention evaluation | False |
| `--ragas-only` | Path to intermediate CSV | None |
| `--batch-size` | Questions per Ragas batch | 10 |
| `--batch-delay` | Delay between Ragas batches (seconds) | 5.0 |

### Evaluation Output

The evaluation script generates:
- **Intermediate CSV**: `*_intermediate.csv` - Contains generated answers and retrieval metrics (saved incrementally)
- **Final CSV**: `ragas_eval_results.csv` - Contains all metrics including Ragas scores

**Columns in final output:**
- `idx`, `question`, `answer`, `ground_truth`
- `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` (Ragas)
- `ndcg`, `ndcg@3`, `ndcg@5`, `recall@3`, `recall@5` (Retrieval)
- `citation_precision`, `ais_recall`, `ais_f1` (Citation)
- `latency_sec`, `num_chunks`, `num_cited` (Performance)

## 🐛 Troubleshooting

### Qdrant Connection Errors

```bash
# Check if Qdrant is running
docker-compose ps

# View Qdrant logs
docker-compose logs qdrant

# Restart Qdrant
docker-compose restart qdrant
```

### Clear Index and Restart

If you encounter indexing issues or want to reprocess documents:

```bash
# Clear the Qdrant index
python clear_index.py

# Restart Qdrant
docker-compose restart qdrant

# Restart Streamlit app (will auto-reingest)
streamlit run app.py
```

**Note**: `clear_index.py` will:
- Delete the Qdrant collection
- Remove processed files metadata
- Allow fresh ingestion on next app startup

### API Rate Limit Errors

If you encounter rate limit errors during evaluation:

```bash
# Increase delay between calls
python evaluation/eval.py --delay 3.0

# Use smaller Ragas batch size
python evaluation/eval.py --ragas-only intermediate.csv --batch-size 5 --batch-delay 15
```

### Timeout Errors During Ragas Evaluation

Ragas evaluation uses batching with automatic retry. If timeouts persist:

```bash
# Use smaller batches with longer delays
python evaluation/eval.py --ragas-only intermediate.csv \
    --batch-size 5 \
    --batch-delay 20
```

The system automatically retries failed batches with exponential backoff (30s → 60s → 120s).

### Missing Documents

If documents aren't appearing:

1. **Check `data/` folder**: Ensure PDFs are in `data/` directory
2. **Check processed files**: Look at `qdrant_metadata/processed_files.json`
3. **Clear and restart**: Run `python clear_index.py` and restart app
4. **Check logs**: Look for errors in Streamlit console output

### Environment Variable Issues

```bash
# Verify .env file exists
ls -la .env

# Check if variables are loaded (in Python)
python -c "from config import Config; print(Config.JINA_API_KEY[:10] if Config.JINA_API_KEY else 'Not set')"
```

## 📊 Access Points

- **Streamlit App**: http://localhost:8501
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Qdrant REST API**: http://localhost:6333
- **Qdrant gRPC API**: http://localhost:6334

## 🔧 Advanced Usage

### Custom Retrieval

```python
from rag_chain import retrieve_only, generate_answer_with_citations

# Retrieval only (no generation)
chunks = retrieve_only("your query", k=5)

# Full RAG pipeline
response = generate_answer_with_citations("your query")
```

### Programmatic Index Clearing

```python
from clear_index import clear_all

# Clear index programmatically
success = clear_all()
```

### Batch Document Processing

```python
from vectorstore import process_all_pdfs_in_folder

# Process all PDFs in a folder
results = process_all_pdfs_in_folder("data/")
print(f"Processed {results['total_files']} files")
```

## 📚 Key Components

### RAG Pipeline (`rag_chain.py`)

1. **Query Processing**: Generates dense and sparse embeddings
2. **Hybrid Retrieval**: Combines dense semantic + sparse keyword search
3. **Reranking**: Uses Jina reranker for precision
4. **Generation**: Groq LLM generates answer with citations
5. **Abstention**: Refuses to answer out-of-scope questions

### Vector Store (`vectorstore.py`)

- Document ingestion and chunking
- Multi-vector embedding generation (Dense + Sparse)
- Qdrant collection management
- Metadata tracking for processed files

### Evaluation (`evaluation/eval.py`)

- **Phase 1**: Incremental answer generation (saves after each question)
- **Phase 2**: Batch Ragas evaluation (with retry logic)
- **Phase 3**: Merge and save final results

## 💡 Best Practices

1. **First Run**: Let the app complete auto-ingestion before querying
2. **Rate Limits**: Use `--delay` flag during evaluation to avoid API limits
3. **Batch Size**: Smaller Ragas batches (5-10) reduce timeout risk
4. **Index Management**: Use `clear_index.py` before major changes
5. **Testing**: Run tests after configuration changes

## 📄 License

MIT

---

**Built with** Qdrant • Jina AI • Groq • Streamlit • LangChain • Ragas
