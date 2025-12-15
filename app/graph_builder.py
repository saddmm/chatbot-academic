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

    # --- PERUBAHAN DIMULAI DARI SINI ---
    
    # 1. Panggil format_docs untuk mengubah List[Document] menjadi String
    #    Ini akan menempelkan URL/Link ke dalam teks konteks agar terbaca LLM
    formatted_context = format_docs(documents)

    # 2. Gunakan Chain standar (Prompt | LLM | Parser)
    #    Kita TIDAK menggunakan create_stuff_documents_chain lagi karena 
    #    kita sudah memformat dokumennya sendiri secara manual di atas.
    rag_chain = rag_prompt | llm | StrOutputParser()

    answer = rag_chain.invoke(
        {
            "question": question,
            "context": formatted_context, # Masukkan string yang sudah ada URL-nya
            "chat_history": message,
            "sources": sources,
        }
    )
    # --- PERUBAHAN SELESAI ---

    print(f"Generated answer: {answer}")
    
    return {"messages": [AIMessage(content=answer)]}

def node_answer_general_chat(state: GraphState, llm, general_chat_prompt) -> str:
    """
    Handle general chat questions that do not require document retrieval.
    """
    general_chat_chain = general_chat_prompt | llm | StrOutputParser()

    # Ambil semua pesan kecuali yang terakhir (pertanyaan saat ini) sebagai history
    chat_history = state['messages'][:-1]
    question = state["messages"][-1].content

    response = general_chat_chain.invoke({
       "chat_history": chat_history,
        "input": question
    })
    
    print(f"Generated general chat response: {response}")
    
    return {"messages": [AIMessage(content=response)]}

def node_classify_question(state: GraphState, llm, classification_prompt) -> str:
    """
    Classify the question to determine if it requires RAG or general chat response.
    """
    question = state["messages"][-1].content
    classification_chain = classification_prompt | llm | StrOutputParser()

    classification_result = classification_chain.invoke({"question": question})
    cleaned_query_type = classification_result.strip().lower()
    
    print(f"Classified question as: {cleaned_query_type}")
    
    # Check for rag_query with underscore or space, or if it contains "rag"
    if "rag_query" in cleaned_query_type or "rag query" in cleaned_query_type or "rag" in cleaned_query_type:
        return {"query_type": "rag_query"}
    else:
        return {"query_type": "general_chat"}
    
def decide_path(state: GraphState) -> str:
    if state["query_type"] == "rag_query":
        print(f"Deciding path for query type: {state['query_type']}")
        return "condense_question"
    else:
        print(f"Deciding path for query type: {state['query_type']}")
        return "generate_answer_general"

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        # 1. Ambil URL Link (Dokumen/Web)
        url = (doc.metadata.get("url") or 
               doc.metadata.get("direct_link") or 
               doc.metadata.get("web_link") or 
               doc.metadata.get("link") or 
               "")
        
        # 2. Ambil URL Gambar (BARU)
        image_url = doc.metadata.get("image_url", "")
        
        source = doc.metadata.get("source", "Dokumen")
        content = doc.page_content
        
        # 3. Susun String Konteks
        doc_str = f"Sumber: {source}\nIsi: {content}"
        
        if url:
            doc_str += f"\nLink Terkait: {url}"
            
        if image_url:
            doc_str += f"\nFoto Fasilitas: {image_url}" 
            
        formatted_docs.append(doc_str)
        
    return "\n\n---\n\n".join(formatted_docs)

def create_graph(llm, retriever, rag_prompt, condense_prompt, classification_prompt, general_chat_prompt ,memory):
    """
    Membuat dan mengompilasi StateGraph LangGraph.
    """

    workflow = StateGraph(GraphState)

    # Tambahkan node-node ke dalam alur kerja
    workflow.add_node("classify_question", lambda state: node_classify_question(state, llm, classification_prompt))
    workflow.add_node("condense_question", lambda state: node_condense_question(state, llm, condense_prompt))
    workflow.add_node("retrieve_documents", lambda state: node_retrieve_documents(state, retriever))
    workflow.add_node("generate_answer_rag", lambda state: node_answer_rag(state, llm, rag_prompt))
    workflow.add_node("generate_answer_general", lambda state: node_answer_general_chat(state, llm, general_chat_prompt))

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
