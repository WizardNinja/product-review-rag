import os
import re
import pandas as pd
from dotenv import load_dotenv
from typing import Any, List, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

RAG_PROMPT_TEMPLATE = """You are a product review analyst helping a business understand customer feedback.

Based on the following customer reviews, answer the user's question. Be specific and cite evidence from the reviews when possible.

CUSTOMER REVIEWS:
{context}

USER QUESTION: {question}

Instructions:
- Provide a clear, actionable answer based on the reviews
- If multiple reviews mention the same point, note that it's a common theme
- If reviews have conflicting opinions, acknowledge both perspectives
- If the reviews don't contain relevant information, say so honestly
- Keep your answer concise but comprehensive

ANSWER:"""


def clean_text(text: str) -> str:
  if not isinstance(text, str):
    return ""

  clean_text = re.sub(r'<[^>]+>', '', text)
  clean_text = re.sub(r'\s+', ' ', clean_text)
  clean_text = clean_text.strip()

  return clean_text


def load_reviews(file_path: str, text_column: str = 'Text', rating_column: str = 'Score', product_id_column: str = 'ProductId') -> pd.DataFrame:
  df = pd.read_csv(file_path)
  df['clean_text'] = df[text_column].apply(clean_text)
  df = df[df['clean_text'].str.len() > 10]
  df = df.drop_duplicates(subset=['clean_text', product_id_column])
  df.rename(columns={
    text_column: 'original_text',
    rating_column: 'rating',
    product_id_column: 'product_id'
  }, inplace=True)

  return df


def create_documents(df: pd.DataFrame) -> List[Document]:
  documents = []

  for idx, row in df.iterrows():
    rating = row.get('rating', 3)
    metadata = {
      'rating': int(rating),
      'word_count': len(row['clean_text'].split()),
      'source_index': idx,
      'product_id': str(row['product_id'])
    }

    documents.append(Document(page_content=row['clean_text'], metadata=metadata))

  return documents



def create_vector_store(documents: List[Document]):
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
  vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db", collection_name="product_reviews")

  return vectorstore


def load_vector_store() -> Chroma:
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
  vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="product_reviews")

  return vectorstore


def search_store(vectorstore: Chroma, query: str, k: int = 5, filter_dict: Optional[dict] = None) -> List[tuple]:
  if filter_dict:
    return vectorstore.similarity_search_with_score(query, k=k, filter=filter_dict)
  else:
    return vectorstore.similarity_search_with_score(query, k=k)

def format_docs(docs: List[Document]) -> str:
  formatted = []
  for doc in enumerate(docs, 1):
    rating = doc[1].metadata.get('rating', 'N/A')
    formatted.append(f"Rating: {rating}/5\n{doc[1].page_content}")
  return "\n\n".join(formatted)

def query_product(question: str, product_id: Optional[str] = None):
  vectorstore = load_vector_store()
  filter_dict = None
  
  if product_id:
    filter_dict = {"product_id": product_id}

  results = search_store(vectorstore, question, k=10, filter_dict=filter_dict)

  if not results:
    return "No reviews found for this product."
    
  docs = [doc for doc, score in results]
  context = format_docs(docs)
  
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
  prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

  chain = prompt | llm | StrOutputParser()
  
  return chain.invoke({"context": context, "question": question})


def query_product_with_sources(question: str, vectorstore: Chroma = None, k: int = 5, product_id: Optional[str] = None) -> Dict[str, Any]:
  if vectorstore is None:
    vectorstore = load_vector_store()

  filter_dict = None
  if product_id:
    filter_dict = {"product_id": product_id}

  results = search_store(vectorstore, question, k=k, filter_dict=filter_dict)

  if not results:
    return {"answer": "No reviews found for this product.", "sources": []}

  docs = [doc for doc, score in results]
  scores = [score for doc, score in results]
  context = format_docs(docs)

  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
  prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

  chain = prompt | llm | StrOutputParser()

  answer = chain.invoke({"context": context, "question": question})

  sources = []
  for doc, score in zip(docs, scores):
    rating = doc.metadata.get('rating', 'N/A')
    sources.append({
      'text': doc.page_content,
      'rating': rating,
      'product_id': doc.metadata.get('product_id'),
      'similarity_score': score
    })

  return {"answer": answer, "sources": sources}
