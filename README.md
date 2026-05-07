# Product Review RAG System

A RAG (Retrieval-Augmented Generation) system that answers natural language questions about product reviews using semantic search and GPT-4o-mini.

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

4. The example reviews_subset.csv was created using the `Reviews.csv` (Amazon Fine Food Reviews dataset) which you can download here https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download

## Using Your Own Reviews

To use reviews for your own products, create a CSV file with these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `Text` | Yes | The review text |
| `Score` | Yes | Rating (1-5) |
| `ProductId` | No | Optional product identifier (for filtering) if uploading reviews for multiple products |

**Example CSV:**
```csv
Text,Score,ProductId
"Great product, works perfectly!",5,PROD001
"Broke after a week. Disappointed.",2,PROD001
"Good value for the price.",4,PROD002
```

Then load it with custom column names if needed:
```python
df = load_reviews(
    'my_reviews.csv',
    text_column='Text',
    rating_column='Score',
    product_id_column='ProductId'
)
```

## Usage

### Building the Vector Store

```python
from product_rag_system import load_reviews, create_documents, create_vector_store

df = load_reviews('reviews_subset.csv')
documents = create_documents(df)
vectorstore = create_vector_store(documents)
```

### Querying Reviews

```python
from product_rag_system import query_product

# Query all products
answer = query_product("What do customers complain about?")

# Query specific product
answer = query_product("What could be better about this product?", product_id="B007JFMH8M")
```

## Tech Stack

- **Vector Store**: ChromaDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: OpenAI GPT-4o-mini
- **Framework**: LangChain
