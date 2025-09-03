from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document as LCDocument
# Updated imports from langchain_openai and langchain
from langchain_openai.chat_models import ChatOpenAI  # Updated import
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings  # Updated import
from langchain.docstore.document import Document as LCDocument

def build_retriever(documents):
    """Builds a retriever using FAISS and OpenAI embeddings."""
    try:
        lc_docs = [LCDocument(page_content=doc.text, metadata=doc.metadata) for doc in documents]
        embedding = OpenAIEmbeddings()  # Embedding model
        vectorstore = FAISS.from_documents(lc_docs, embedding)  # Build FAISS index from documents
        return vectorstore.as_retriever(search_kwargs={'k': 3})  # Return top 3 most relevant results
    except Exception as e:
        print(f"Error building retriever: {e}")
        raise

def build_qa_chain(retriever):
    """Creates a QA chain using a retriever and LLM model."""
    try:
        llm = ChatOpenAI(temperature=0)  # Use a zero-temperature model for deterministic results
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    except Exception as e:
        print(f"Error building QA chain: {e}")
        raise
