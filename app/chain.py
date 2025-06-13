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
    messages: Annotated[Sequence[BaseMessage], operator.add]
    document: List[Document]
    sources: str
    query_type: str


def node_condense_question(state: GraphState, llm, condense_prompt) -> str:
    """
    Condense the question from the state.
    """
    chat_history = state["messages"][:-1]
    new_question = state["messages"][-1].content

    if not chat_history:
        return {"question": new_question}
    
    condense_question_chain = condense_prompt | llm | StrOutputParser()

    condensed_question = condense_question_chain.invoke(
        {
            "chat_history": chat_history,
            "question": new_question,
        }
    )
    print(f"Condensed question: {condensed_question}")
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
    
    print(f"Retrieved {len(documents)} documents for question: {question}")
    return {"document": documents, "sources": sources_str}

def node_answer_rag(state: GraphState, llm, rag_prompt) -> str:
    """
    Answer the question using the retrieved documents.
    """
    question = state["question"]
    documents = state["document"]
    message = state["messages"]
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

    print(f"Generated answer: {answer}")
    
    return {"messages": [AIMessage(content=answer)]}

def node_answer_general_chat(state: GraphState, llm) -> str:
    """
    Handle general chat questions that do not require document retrieval.
    """
    question = state["messages"][-1].content
    response = llm.invoke({"question": question})
    
    print(f"Generated general chat response: {response}")
    
    return {"messages": [AIMessage(content=response)]}

def node_classify_question(state: GraphState, llm, classification_prompt) -> str:
    """
    Classify the question to determine if it requires RAG or general chat response.
    """
    question = state["messages"][-1].content
    classification_chain = classification_prompt | llm | StrOutputParser()

    classification_result = classification_chain.invoke({"question": question})
    
    print(f"Classified question as: {classification_result}")
    
    if classification_result == "rag_query":
        return {"query_type": "rag_query"}
    elif classification_result == "general_chat":
        return {"query_type": "general_chat"}
    else:
        raise ValueError(f"Unknown query type: {classification_result}")
    
def decide_path(state: GraphState) -> str:
    if state["query_type"] == "rag_query":
        print(f"Deciding path for query type: {state['query_type']}")
        return "condense_question"
    else:
        print(f"Deciding path for query type: {state['query_type']}")
        return "generate_answer_general"

def create_graph(llm, retriever, rag_prompt, condense_prompt, classification_prompt ,memory):
    """
    Membuat dan mengompilasi StateGraph LangGraph.
    """

    workflow = StateGraph(GraphState)

    # Tambahkan node-node ke dalam alur kerja
    workflow.add_node("classify_question", lambda state: node_classify_question(state, llm, classification_prompt))
    workflow.add_node("condense_question", lambda state: node_condense_question(state, llm, condense_prompt))
    workflow.add_node("retrieve_documents", lambda state: node_retrieve_documents(state, retriever))
    workflow.add_node("generate_answer_rag", lambda state: node_answer_rag(state, llm, rag_prompt))
    workflow.add_node("generate_answer_general", lambda state: node_answer_general_chat(state, llm))

    # Tentukan alur kerjanya
    workflow.set_entry_point("classify_question")

    workflow.add_conditional_edges(
        "classify_question",
        decide_path,
        {
            "condense_question": "condense_question",
            "generate_answer_general": "generate_answer_general"
        }
    )

    workflow.add_edge("condense_question", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_answer_rag")
    workflow.add_edge("generate_answer_rag", END)
    workflow.add_edge("generate_answer_general", END)

    # Kompilasi alur kerja menjadi objek yang bisa dijalankan
    graph = workflow.compile(checkpointer=memory)
    if not graph:
        raise ValueError("Failed to compile the graph. Please check the workflow configuration.")
    return graph
