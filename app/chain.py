from ast import Str
from typing import Any, Dict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def create_rag_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever, prompt: ChatPromptTemplate):
    """
    Create a Retrieval Augmented Generation (RAG) chain using the provided LLM and vector store.

    Args:
        llm (BaseLanguageModel): The language model to use for generating responses.
        vectorstore (FAISS): The vector store containing the indexed documents.

    Returns:
        RetrievalQA: The RAG chain ready for use.
    """
    def retrieve_format_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        retrieved_docs = retriever.get_relevant_documents(question)
        context, sources = format_retrieved_docs(retrieved_docs)
        return {
            "context": context,
            "sources": sources,
            "question": question
        }

    rag_chain_from_docs = (
        RunnableLambda(retrieve_format_context)
        | prompt
        | llm
        | StrOutputParser()
    )
        
    
    return rag_chain_from_docs