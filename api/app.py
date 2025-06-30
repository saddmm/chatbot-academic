import os
from pydoc import doc
from pyexpat import model
import uuid
from flask import Flask, abort, make_response, request, jsonify
from httpx import get
from app.chain import create_graph
from app.document_processor import process_document_for_rag, process_urls_for_rag
from app.llm_config import get_embedding, get_groq_llm, get_llm
from app.prompt import CONDENS_QUESTION_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE
from app.vectorstore import get_or_create_vector_store
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
import sqlite3

set_llm_cache(SQLiteCache(database_path=".langchain_cache.sqlite"))

app = Flask(__name__)

try:
    llm = get_groq_llm(model_name="llama3-8b-8192", temperature=0.1)
    # llm = get_llm(model_name="llama3.1:8b", temperature=0.5)
    embedding_model = get_embedding()

    # document_chunks = process_document_for_rag(url_list_file_path="urls.txt", chunk_size=1000, chunk_overlap=150)
    # print(f"Jumlah dokumen yang diproses: {document_chunks}")s
    vector_store = get_or_create_vector_store(
        # documents=document_chunks,
        embedding_model=embedding_model,
    )
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 10}  # Mengambil 5 dokumen relevan
    )
    sqlite_conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(sqlite_conn)
    print("memory initialized with SqliteSaver.")
    graph = create_graph(
        llm=llm,
        retriever=retriever,
        rag_prompt=RAG_PROMPT_TEMPLATE,
        condense_prompt=CONDENS_QUESTION_PROMPT_TEMPLATE,
        memory=memory
    )
    print("Komponen LLM dan Vector Store berhasil diinisialisasi.")
except Exception as e:
    print(f"Error saat menginisialisasi LLM atau Vector Store: {e}")
    graph = None

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()
    
    question = data.get("question")
    session_id = request.cookies.get("session_id")


    if not session_id:
        session_id = str(uuid.uuid4()) # 1 day expiration
    
    if not graph:
        abort(500, "Graph is not initialized. Please check the server logs for details.")
    

    try:
        config = {"configurable": {"thread_id": session_id}}

        input_data = {
            "messages": [HumanMessage(content=question)],
            "question": question,
        }

        print(f"Received question: {question} with session_id: {session_id}")

        final_state = graph.invoke(
            input_data,
            config,
        )

        answer = final_state["messages"][-1].content
        response_data = {
            "answer": answer,
            "session_id": session_id
        }
        response = make_response(jsonify(response_data))
        if not request.cookies.get("session_id"):
            response.set_cookie("session_id", session_id, max_age=60*60*24)
        print(response)
        return response
    except Exception as e:
        print(f"Error during graph invocation: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)  # Set debug=True for development purposes
