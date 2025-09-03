# Import necessary components from LlamaIndex
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.openai import OpenAIEmbeddings

def create_documents(df):
    """Creates documents from email data to be indexed."""
    docs = []
    for _, row in df.iterrows():
        content = f"Subject: {row['subject']}\nBody: {row['message']}"
        docs.append(Document(text=content, metadata={'label': row['label']}))
    return docs

def build_index(documents):
    """Builds an index using the LlamaIndex framework."""
    embed = LangchainEmbedding(OpenAIEmbeddings())  # Embedding model to convert text into vectors
    index = VectorStoreIndex.from_documents(documents, embed_model=embed)  # Create the index
    return index
