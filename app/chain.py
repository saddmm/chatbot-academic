from typing import Annotated, List, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
import operator
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph, END
import os

class GraphState(TypedDict):
    question: str
    message: Annotated[Sequence[BaseMessage], operator.add]
    document: List[Document]
    sources: str


def node_condense_question(state: GraphState, llm, condense_prompt) -> str:
    """
    Condense the question from the state.
    """
    chat_history = state["message"][:-1]
    new_question = state["message"][-1].content

    if not chat_history:
        return {"question": new_question}
    
    condense_question_chain = condense_prompt | llm | StrOutputParser()

    condensed_question = condense_question_chain.invoke(
        {
            "chat_history": chat_history,
            "question": new_question,
        }
    )
    return {"question": condensed_question}

def node_retrieve_documents(state: GraphState, retriever) -> GraphState:
    """
    Retrieve documents based on the question in the state.
    """
    question = state["question"]
    documents = retriever.invoke(question)
    if not documents:
        sources_str = "Tidak ada sumber dokumen yang spesifik."
    else:
        unique_sources = set()
        for doc in documents:
            # Ambil nama file dari path 'source' di metadata
            source = doc.metadata.get('source', 'Tidak diketahui')
            unique_sources.add(os.path.basename(source))
        sources_str = "\n".join([f"- {s}" for s in sorted(list(unique_sources))])
    
    return {"document": documents, "sources": sources_str}

def node_answer_question(state: GraphState, llm, rag_prompt) -> str:
    """
    Answer the question using the retrieved documents.
    """
    question = state["question"]
    documents = state["document"]
    message = state["message"]
    sources = state["sources"]

    rag_chain = create_stuff_documents_chain(llm, rag_prompt)
    answer = rag_chain.invoke(
        {
            "question": question,
            "context": documents,
            "chat_history": message,
            "sources": sources,
        }
    )
    
    return {"message": [AIMessage(content=answer)]}

def create_graph(llm, retriever, rag_prompt, condense_prompt):
    """
    Membuat dan mengompilasi StateGraph LangGraph.
    """
    workflow = StateGraph(GraphState)

    # Tambahkan node-node ke dalam alur kerja
    workflow.add_node("condense_question", lambda state: node_condense_question(state, llm, condense_prompt))
    workflow.add_node("retrieve_documents", lambda state: node_retrieve_documents(state, retriever))
    workflow.add_node("generate_answer", lambda state: node_answer_question(state, llm, rag_prompt))

    # Tentukan alur kerjanya
    workflow.set_entry_point("condense_question")
    workflow.add_edge("condense_question", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Kompilasi alur kerja menjadi objek yang bisa dijalankan
    graph = workflow.compile()
    
    return graph
